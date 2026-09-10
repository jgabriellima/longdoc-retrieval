# longdoc-retrieval

Recuperação de evidências em documentos longos e estruturalmente complexos,
sem embeddings e sem nunca colocar o documento inteiro no prompt de um LLM.

## O problema

Documentos longos (contratos, processos administrativos, petições
governamentais) não cabem em um prompt, e embeddings nem sempre são uma
opção — ou o melhor caminho — para o que deveria ser recuperação exata:
uma cláusula específica, um número específico, um termo definido
específico. Este projeto responde a uma pergunta mais estreita: dado um
documento longo e uma pergunta sobre ele, como encontrar a evidência de
apoio de forma determinística e barata, e só então trazer um LLM para
decidir *o que procurar em seguida* e *se o que já foi encontrado é
suficiente* — nunca para guardar o documento inteiro na cabeça.

## Como funciona

Duas camadas, empilhadas.

### 1. Recuperação determinística

`domain/`, `ingestion/`, `indexes/`, `retrieval/`, `api/` — nenhum LLM neste caminho. Um documento é
normalizado e parseado em uma árvore estrutural (títulos/seções, detecção
bilingue EN/PT-BR — ARTICLE/SECTION e também ARTIGO/SEÇÃO/CAPÍTULO —
caindo para blocos normalizados quando nenhuma estrutura é detectável),
depois fragmentado em unidades de recuperação de 150–400 tokens ao longo
de fronteiras de parágrafo — a unidade que de fato é indexada. Dois
índices ficam por cima: um índice esparso BM25 (SQLite FTS5 por padrão,
Tantivy como alternativa plugável — ver abaixo) para consultas em
linguagem natural, e um scan direto por regex para identificadores (datas,
valores monetários, referências legislativas, CPFs/CNPJs, números de
contrato) — ranking lexical não se aplica a algo que ou casa exatamente,
ou não casa. `RetrievalService` expõe exatamente 7 métodos sobre isso:
`document_outline`, `search`, `find_exact`, `inspect_node`, `expand_node`,
`read_node`, `read_range` — nenhum deles devolve o documento inteiro por
padrão.

### 2. Loop agentico de recuperação

![Grafo de recuperação compilado](graph.png)

Dado `(document_id, question)`, a camada `graph/` — construída sobre
[LangGraph](https://github.com/langchain-ai/langgraph) — executa um grafo
de estado compilado nesta sequência, voltando para `execute_searches`
quando uma passada não basta:

| Nó | Chamadas LLM | O que faz |
|---|---|---|
| `understand_request` | 0 | Busca um `document_outline` de profundidade 2 e o renderiza como uma árvore indentada de títulos — a única visão da estrutura do documento que o planner recebe, nunca o texto completo. |
| `plan_retrieval` | 1 | Transforma a pergunta (+ outline, + o que a iteração anterior disse que faltava) em um `RetrievalPlan`: um objetivo, `concepts` lexicais, `exact_terms` para busca de identificadores, `structural_hints` e uma lista `queries`. Se a chamada de structured-output falhar ou for inválida, cai para um plano vazio em vez de crashar ou inventar um — o próximo nó simplesmente não encontra nada, e a suficiência reporta corretamente que falta informação. |
| `execute_searches` | 0 | Concatena `queries` + `concepts` (ambos esparsos) + `exact_terms` (exatos) *nessa ordem*, deduplica por (método, query), trunca em `max_queries_per_iteration` e executa tudo em paralelo via `asyncio.gather`. Como `queries` vem primeiro, um plano com mais `queries` do que o orçamento permite pode excluir `concepts`/`exact_terms` por completo naquela iteração. |
| `merge_candidates` | 0 | Deduplica por `candidate_id`, mescla intervalos sobrepostos no mesmo nó (exact > structural > sparse > expanded vence em conflito), depois agrupa por seção de topo e faz round-robin entre os grupos (ordenados por score lexical dentro de cada um) até `max_candidates_for_llm_evaluation` — assim dez hits de um único parágrafo não expulsam outras seções relevantes. |
| `evaluate_candidates` | 1 (em lote) | Uma única chamada julga *todos* os candidatos fundidos de uma vez, a partir de prévias curtas, nunca do texto completo. Se o modelo omitir um candidato da resposta, esse é preenchido como `possibly_relevant/should_read=false` em vez de ser descartado em silêncio; se a chamada inteira falhar, todo candidato assume `should_read=true` — um avaliador instável degrada custo/amplitude, nunca correção. |
| `read_evidence` → `update_evidence_ledger` | 0 | Abre em texto completo, via `read_range`, apenas os candidatos com `should_read=true`. O ledger só funde um item novo em um existente quando o ponto de junção parece um corte real no meio da frase (sem pontuação de fechamento imediatamente antes) — não só porque duas unidades de recuperação acontecem de ser adjacentes, o que quase sempre são. Limitado por `max_evidence_items`/`max_evidence_tokens`. |
| `evaluate_sufficiency` | 1 por item de evidência + 1 em lote | Cada item de evidência é julgado em uma chamada isolada (vendo os outros só como contexto de apoio não invalidante, p.ex. para resolver um rótulo de papel definido em outro trecho) — um prompt compartilhado pedindo ao modelo para julgar os itens "independentemente" não foi confiável na prática; ruído de contradição de um item suprimia uma resposta correta em outro. Uma chamada em lote sobre tudo fornece `missing_information`/`contradictions`/`recommended_queries` e uma segunda tentativa independente de extração. Em qualquer dos casos, o `answer_excerpt` citado é verificado em código como substring real da evidência antes de `sufficient` poder virar `true` — a afirmação do próprio modelo nunca é confiável às cegas. |
| *(`should_stop`, chamado aqui)* | 0 | Uma função pura (`graph/budgets.py`), checada nesta ordem exata, parando na primeira condição verdadeira: `sufficient` → `max_iterations` → `evidence_token_budget` → `llm_call_budget` → `time_budget` → `no_new_information` (contagem de evidências inalterada por 2 iterações seguidas). O roteamento e o relatório final leem o mesmo valor computado, então nunca podem discordar sobre por que o loop parou. |
| `refine_strategy` | 0 | Só é alcançado se não for para parar: monta o próximo `RetrievalPlan` direto a partir de `recommended_queries`, reusando os `exact_terms`/`structural_hints` anteriores — sem nova chamada LLM de planejamento. |
| `build_evidence_package` | 0 | Nó terminal. `status` é `sufficient` se o loop concluiu positivamente, `partial` se parou com alguma evidência mas sem conclusão, `insufficient` se não há evidência alguma — nunca fabricado em nenhum dos casos. |

Todos os orçamentos do loop acima (`max_iterations=6`,
`max_queries_per_iteration=8`, `max_candidates_per_query=20`,
`max_candidates_for_llm_evaluation=30`, `max_evidence_items=20`,
`max_evidence_tokens=20_000`, `max_total_llm_calls=30`,
`max_time_seconds=240.0`) vivem em `RetrievalConfig`, não hardcoded no
grafo.

O invariante que vale nas duas camadas: **evidência nunca é fabricada**.
Se o loop não encontrar algo, ele diz isso (`status="insufficient"`) com o
que ainda falta, em vez de chutar. O mecanismo anti-alucinação não é
"perguntar ao modelo se ele tem certeza" — é verificação literal de
substring em código: o modelo precisa copiar o texto exato que responde à
pergunta, e essa cópia é checada contra o documento real antes de ser
aceita como evidência.

## Início rápido

```bash
pip install -e ".[agent]"          # camada determinística + o loop LangGraph
cp .env.example .env               # adicionar ANTHROPIC_API_KEY ou OPENAI_API_KEY
python main.py                     # ingerir um contrato de exemplo, fazer uma pergunta
```

```bash
python main.py --document path/to/file.txt --question "..."
```

`RetrievalConfig.from_env()` escolhe Anthropic (Claude Sonnet 5 para
planejamento/suficiência, Claude Haiku 4.5 para avaliação de candidatos)
se `ANTHROPIC_API_KEY` estiver definida; caso contrário, os modelos
equivalentes da OpenAI.

## Estrutura do projeto

```
src/longdoc_retrieval/
├── domain/       # contratos de dados puros: Document, DocumentNode, RetrievalCandidate, Evidence, ...
├── ingestion/    # normalize -> parse structure -> build nodes -> chunk into retrieval units
├── indexes/      # armazenamento SQLite (árvore estrutural + FTS5) e a alternativa Tantivy
├── retrieval/    # search(), find_exact(), read_node()/read_range()
├── api/          # RetrievalService - a fachada de 7 métodos sobre o restante
├── graph/        # o loop agentico LangGraph: planner, search, fusion,
│                 # evaluator, reader, sufficiency, budgets, router
└── config.py     # RetrievalConfig - todo orçamento do loop e o nome de modelo por componente
```

Só `pydantic` é dependência obrigatória; a camada de recuperação
determinística não importa mais nada. O loop agentico (`graph/`) precisa
do extra `[agent]` (LangGraph + um SDK de provedor de modelo). Ver
`VERSIONS.md` para a matriz completa de compatibilidade, verificada no
PyPI.

## Backends de busca: SQLite FTS5 vs Tantivy

O backend padrão de recuperação esparsa é SQLite FTS5 (ranqueado por BM25,
zero dependências extras). Tem uma limitação real para texto que não é
inglês: o tokenizer não faz stemming, então uma query por "total" nunca
casa com um documento que só contém uma forma flexionada como "totaling"
(ou, em português, "totaliza" vs "total"). Testes contra documentos reais
em PT-BR mostraram isso como um gap genuíno de recall, não hipotético.

[Tantivy](https://github.com/quickwit-oss/tantivy-py) (um motor de busca em
Rust com bindings Python) foi adicionado como segunda implementação da
mesma interface `SparseBackend` especificamente para testar se um motor
consciente de stemmer fecha esse gap — `pip install -e ".[tantivy]"`,
depois `RetrievalService(conn, sparse=TantivySparseRetriever(conn))`. É
uma troca de verdade, não um wrapper: os dois backends implementam o
mesmo contrato `index_units`/`search`, então nada acima de
`retrieval/search.py` precisa saber qual está em uso.

O que uma comparação A/B em documentos reais mostrou:

- **Acurácia**: registrar um stemmer Snowball de português no lado Tantivy
  resolveu misses de recall que o lado SQLite FTS5 de fato tinha (a
  evidência de que o LLM precisava existia no documento, mas a busca
  lexical nunca a trouxe à tona) — sem introduzir novos falsos positivos.
  Uma contagem ingênua de "sufficient/partial" fazia os dois backends
  parecerem próximos; verificar *qual fato específico* fundamentava cada
  resposta (não só o status reportado) revelou um modo de falha que nem a
  contagem ingênua nem a confiança do próprio modelo capturaram: responder
  a partir de uma cláusula real, porém errada, quando a correta nunca foi
  recuperada. Esse modo de falha apareceu só no lado SQLite nesta
  avaliação.
- **Latência**: Tantivy é mensuravelmente mais lento na própria camada de
  recuperação (cerca de 2x na média, com uma cauda mais longa — em grande
  parte porque esta implementação reconstrói o searcher/gerador de
  snippets a cada chamada em vez de cacheá-los) e o custo único de
  construção do índice é uma ordem de grandeza maior. Nenhum dos dois
  importa na prática aqui: a latência ponta a ponta é dominada pelas
  chamadas LLM (dezenas de segundos) por 2–3 ordens de grandeza sobre a
  recuperação (dezenas de milissegundos), então a escolha do backend
  esparso não move a latência percebida pelo usuário.

Saldo: o gap de stemming é real e vale a pena corrigir, e corrigi-lo não
custa nada que um usuário notaria. Os dois backends acompanham o projeto;
qual é o "padrão" é uma escolha de configuração, não arquitetural.

## Testes

```bash
pip install -e ".[dev,agent,tantivy]"
pytest tests -q       # unit + integration, sem chamadas de rede (dublê fake de LLM)
ruff check src tests main.py
mypy src
```

Os testes de integração executam o **grafo compilado completo** contra
documentos reais (3 contratos CUAD de
[LegalBench-RAG](https://github.com/ZeroEntropy-AI/legalbenchrag), em
`examples/legalbench_rag/` — ver `CITATION.md` ali para proveniência) com
um LLM fake scriptado, de modo que rastreabilidade de evidência,
término do loop e concordância roteamento/motivo-de-parada são checados
contra offsets reais do documento, sem gastar uma chamada de API de
verdade.

`main.py` é o único script que faz chamadas reais de LLM — útil para um
sanity check manual dos prompts de planner/evaluator/sufficiency contra
comportamento do mundo real antes de confiar neles adiante, mas não faz
parte da suíte automatizada.
