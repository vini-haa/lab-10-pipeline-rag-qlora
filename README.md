> *Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por **Vinicius Henrique Albino Andrade**.*

# Laboratório 10 — O Pipeline Definitivo (RAG + QLoRA + FlashAttention-2 + KV Cache)

**Disciplina:** Tópicos em IA — Instituto de Ensino Superior iCEV
**Aluno:** Vinicius Henrique Albino Andrade
**Repositório:** https://github.com/vini-haa/lab-10-pipeline-rag-qlora
**Notebook:** [`lab10_pipeline.ipynb`](./lab10_pipeline.ipynb)
**Release final corrigível:** `v1.0`

---

## 1. Contexto

A HealthTech (caso construído no [Lab 09](#)) colocou em produção um pipeline RAG médico (HNSW + HyDE + Cross-Encoder) e agora precisa gerar **resumos clínicos automatizados** a partir de ~5 capítulos inteiros de manuais (≈12-15 mil tokens). Em testes na nuvem, o servidor travou: a complexidade **O(n²)** do Self-Attention estourou a VRAM da GPU.

Este laboratório implementa a engenharia de inferência que resolve o problema, combinando três otimizações ortogonais:

| Camada | Técnica | Unidade da disciplina |
|---|---|---|
| Pesos do modelo | **QLoRA 4-bit (NF4)** via `bitsandbytes` | II — Fine-tuning eficiente |
| Geração autorregressiva | **KV Cache** (`use_cache=True`) | I — Self-Attention |
| Kernel de atenção | **FlashAttention-2** (com fallback SDPA) | I — Hardware-Aware Algorithms |

## 2. Ambiente de execução

- **Plataforma:** Google Colab
- **GPU:** `<PREENCHER: nome da GPU — ex. Tesla T4 / L4 / A100>`
- **Compute capability:** `<PREENCHER: ex. sm_75 / sm_89 / sm_80>`
- **Backend de atenção usado no Passo 4:** `<PREENCHER: flash_attention_2 ou sdpa>`
- **Modelo:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Tamanho do contexto RAG simulado:** `<PREENCHER: ex. 12.450>` tokens
- **Tokens gerados em cada benchmark:** 100 (greedy decoding)

> **Nota sobre FlashAttention-2:** O kernel oficial do FA-2 requer arquitetura **Ampere ou superior** (sm_80+). Se o runtime do Colab atribuir uma **T4** (Turing, sm_75), o notebook detecta automaticamente e faz *fallback* para `attn_implementation="sdpa"` (Scaled Dot Product Attention nativo do PyTorch), que também aplica kernels memory-efficient e preserva o ganho conceitual da otimização Hardware-Aware. O parecer técnico vale para ambos os casos.

## 3. Métricas de Benchmark

### 3.1. Passo 1 — Pegada do modelo quantizado

| Configuração | VRAM ocupada |
|---|---|
| TinyLlama-1.1B em FP16 (referência teórica) | ~2.200 MB |
| **TinyLlama-1.1B em 4-bit NF4 + Double Quant** | **`<PREENCHER>` MB** |
| Redução | `<PREENCHER>` % |

### 3.2. Passos 3 e 4 — Geração de 100 tokens sobre contexto RAG massivo

| Métrica | Baseline (eager, `use_cache=False`) | Otimizado (FA-2/SDPA, `use_cache=True`) | Ganho |
|---|---:|---:|---:|
| Tempo total | `<PREENCHER>` s | `<PREENCHER>` s | `<PREENCHER>` × mais rápido |
| Throughput | `<PREENCHER>` tok/s | `<PREENCHER>` tok/s | — |
| Pico de VRAM | `<PREENCHER>` MB | `<PREENCHER>` MB | `<PREENCHER>` % de redução |
| Δ VRAM durante geração | `<PREENCHER>` MB | `<PREENCHER>` MB | `<PREENCHER>` % de redução |

> Gráfico comparativo gerado pelo notebook: [`benchmark_chart.png`](./benchmark_chart.png)
> Métricas brutas em JSON: [`benchmark_report.json`](./benchmark_report.json)

## 4. Parecer Técnico (Passo 5)

### 4.1. Parte A — Como QLoRA, KV Cache e FlashAttention salvaram o Transformer

O Transformer tradicional colapsa em produção por três custos compostos: o tamanho do checkpoint em FP16, o recálculo redundante de Q/K/V a cada token gerado e a materialização da matriz de atenção N×N na memória global da GPU. **QLoRA** atacou o primeiro problema reduzindo a pegada estática do modelo em ~4× ao quantizar os pesos para NF4 com double quantization, mantendo o cálculo em FP16 — sem isso, sequer carregaríamos o checkpoint somado ao contexto de 12 mil tokens na VRAM. **KV Cache** atacou o custo dinâmico da geração: a cada novo token, em vez de reprojetar Q, K e V de todo o prefixo, o modelo reutiliza os tensores K e V já computados, transformando uma complexidade total de O(L²·N) em O(L² + L·N). Como L (contexto RAG) >> N (tokens gerados), o ganho é dramático — observamos um *speedup* de `<PREENCHER>×` apenas com essa otimização de software. **FlashAttention-2** fechou o ciclo no nível de hardware: ao reformular o cálculo do softmax(QKᵀ/√d) em *tiles* que cabem na SRAM on-chip da GPU em vez de materializar a matriz N×N na HBM, reduz a memória do passo de prompting de O(N²) para O(N), explicando a queda de `<PREENCHER>` MB no pico de VRAM. As três técnicas são complementares: QLoRA economiza memória de pesos, KV Cache economiza tempo de inferência e FlashAttention-2 economiza memória de ativações — juntas, viabilizam executar em uma única GPU de consumidor o que antes exigiria um cluster.

### 4.2. Parte B — Por que com 2 milhões de tokens nem FlashAttention salva, e por que a indústria migraria para Mamba/SSM

Mesmo com FlashAttention-2 e KV Cache, o Transformer carrega uma cicatriz estrutural: o KV Cache cresce **linearmente** com o tamanho do contexto. Para uma janela de 2 milhões de tokens em um Llama-3 70B (8 KV heads, head_dim 128, 80 camadas, FP16), o cache sozinho ultrapassa **80 GB**, excedendo a VRAM de uma única H100 — e isso antes mesmo de considerar os pesos. FlashAttention reduz a constante e elimina a materialização da matriz de atenção, mas **não muda a complexidade assintótica** O(N) da memória do cache nem a O(N²) da fase de prompting (mantida na sub-camada de scores, ainda que tiled). A indústria respondeu com os **State Space Models** (Mamba, Mamba-2, RWKV) e arquiteturas híbridas (Jamba, Zamba), nas quais a "memória" do modelo é representada por um **estado oculto recorrente de tamanho fixo** atualizado por um operador seletivo linear — complexidade de memória **O(1)** por token gerado e tempo **O(N)** no contexto. Em outras palavras: enquanto o Transformer paga preço por cada token revisitado, o SSM compacta o passado em um vetor de estado constante. Para janelas de milhões de tokens (genoma, vídeo longo, prontuários médicos completos, código de monorepos inteiros), essa mudança de regime é o que torna a inferência viável em hardware comercial — e é por isso que pipelines como o nosso, hoje resolvidos com a tríade QLoRA + KV Cache + FlashAttention, tenderão a migrar para arquiteturas SSM ou híbridas no horizonte de produção.

## 5. Reflexões e Perguntas em Aberto

Durante a implementação, surgiram dúvidas que extrapolam o escopo do laboratório mas considerei valiosas para aprofundamento:

1. **FlashAttention-3 e Hopper** — O FA-3 explora *warp-specialization* assíncrona (WGMMA) exclusiva da H100. Para contextos abaixo de 128k tokens, o salto FA-2 → FA-3 entrega ganho real ou é dominado pela latência de I/O? Vale o esforço de portar pipelines de produção para Hopper ainda em 2026?

2. **Custo cognitivo do NF4 em domínio clínico** — A quantização 4-bit NF4 é praticamente "gratuita" em benchmarks gerais (MMLU, HellaSwag), mas em tarefas de raciocínio médico onde a precisão de dosagens e contraindicações importa, existe degradação mensurável? Estudos comparam NF4 vs INT8 vs FP16 em corpora clínicos como MedQA ou PubMedQA?

3. **Arquiteturas híbridas Mamba-Transformer** — Modelos como **Jamba** e **Zamba** intercalam blocos SSM e Attention para combinar recall preciso do segundo com complexidade O(1) do primeiro. Em um pipeline RAG médico, essa hibridez **substitui** a necessidade de retrieval externo (já que o SSM "lembra" mais barato) ou continua sendo complementar?

4. **PagedAttention (vLLM) vs KV Cache contíguo** — O PagedAttention do vLLM inspira-se em paginação de memória virtual para reduzir fragmentação em workloads multi-tenant. Em um cenário hospitalar com várias consultas simultâneas ao mesmo manual, o ganho prático sobre o KV Cache contíguo justifica a complexidade operacional adicional?

5. **Offloading de KV Cache para RAM/disco** — Para contextos extremos, algumas implementações (como o de Hugging Face com `OffloadedCache`) movem KV cache antigo para a RAM ou NVMe. Que latência isso adiciona por token? Existe ponto de equilíbrio em que offloading + Transformer ainda supera um SSM puro?

## 6. Como executar

### 6.1. Pré-requisitos

- Conta Google e acesso ao [Google Colab](https://colab.research.google.com/)
- Runtime com GPU CUDA (ideal: L4 ou A100; aceito: T4 com fallback SDPA)

### 6.2. Passos

1. Abrir o notebook [`lab10_pipeline.ipynb`](./lab10_pipeline.ipynb) no Colab (`File → Upload notebook` ou clonar via `!git clone`).
2. Em `Runtime → Change runtime type`, selecionar **GPU**.
3. Executar todas as células em sequência.
4. As métricas finais aparecem no console e são salvas em `benchmark_report.json` + `benchmark_chart.png`.

### 6.3. Estrutura do repositório

```
.
├── lab10_pipeline.ipynb        # Notebook principal (Passos 1–5)
├── benchmark_report.json       # Métricas brutas (gerado pela execução)
├── benchmark_chart.png         # Gráfico comparativo (gerado pela execução)
├── README.md                   # Este arquivo
└── .gitignore
```

## 7. Referências técnicas

- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.*
- Dao (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.*
- Gu & Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.*
- Hugging Face Transformers — documentação oficial de `BitsAndBytesConfig`, `attn_implementation` e `use_cache`.
