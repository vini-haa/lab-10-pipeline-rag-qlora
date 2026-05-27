> *Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por **Vinicius Henrique Albino Andrade**.*

# Laboratório 10 — O Pipeline Definitivo (RAG + QLoRA + FlashAttention-2 + KV Cache)

**Disciplina:** Tópicos em IA — Instituto de Ensino Superior iCEV
**Aluno:** Vinicius Henrique Albino Andrade
**Repositório:** https://github.com/vini-haa/lab-10-pipeline-rag-qlora
**Notebook:** [`lab10_pipeline.ipynb`](./lab10_pipeline.ipynb)
**Release final corrigível:** `v1.0`

---

## 1. Contexto

A HealthTech (caso construído no Lab 09) colocou em produção um pipeline RAG médico (HNSW + HyDE + Cross-Encoder) e agora precisa gerar **resumos clínicos automatizados** a partir de ~5 capítulos inteiros de manuais (≈10-15 mil tokens). Em testes na nuvem, o servidor travou: a complexidade **O(n²)** do Self-Attention estourou a VRAM da GPU.

Este laboratório implementa a engenharia de inferência que resolve o problema, combinando três otimizações ortogonais:

| Camada | Técnica | Unidade da disciplina |
|---|---|---|
| Pesos do modelo | **QLoRA 4-bit (NF4)** via `bitsandbytes` | II — Fine-tuning eficiente |
| Geração autorregressiva | **KV Cache** (`use_cache=True`) | I — Self-Attention |
| Kernel de atenção | **FlashAttention-2** (com fallback SDPA) | I — Hardware-Aware Algorithms |

## 2. Ambiente de execução

- **Plataforma:** Google Colab
- **GPU:** Tesla T4 (15 GB VRAM)
- **Compute capability:** sm_75 (Turing)
- **Backend de atenção usado no Passo 4:** `sdpa` (fallback documentado — ver nota abaixo)
- **Modelo:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Tamanho do contexto RAG simulado:** **9.622 tokens**
- **Tokens gerados em cada benchmark:** 100 (greedy decoding)

> **Nota sobre FlashAttention-2:** O kernel oficial do FA-2 requer arquitetura **Ampere ou superior** (sm_80+). Como o runtime do Colab atribuiu uma **Tesla T4** (Turing, sm_75), o notebook detectou automaticamente e fez *fallback* para `attn_implementation="sdpa"` (Scaled Dot Product Attention nativo do PyTorch). O SDPA também aplica kernels memory-efficient e preserva o ganho conceitual da otimização Hardware-Aware (não materialização da matriz de atenção N×N na HBM). O parecer técnico abaixo vale para ambos os casos.

## 3. Métricas de Benchmark

### 3.1. Passo 1 — Pegada do modelo quantizado

| Configuração | VRAM ocupada |
|---|---|
| TinyLlama-1.1B em FP16 (referência teórica) | ~2.200 MB |
| **TinyLlama-1.1B em 4-bit NF4 + Double Quant** | **735,53 MB** |
| **Redução conquistada pelo QLoRA** | **~66,6 %** (~3× menor) |

### 3.2. Passos 3 e 4 — Geração de 100 tokens sobre contexto RAG massivo

| Métrica | Baseline (eager, `use_cache=False`) | Otimizado (SDPA, `use_cache=True`) | Resultado |
|---|---:|---:|---:|
| Tempo total | **180,33 s** | **6,89 s** | **26,19× mais rápido** ⚡ |
| Throughput | 0,55 tok/s | 14,52 tok/s | +2.518 % |
| Pico de VRAM | 1.212,68 MB | 1.421,04 MB | +17,2 % ⚠️ |
| Δ VRAM gerado | 477,00 MB | 675,71 MB | +41,7 % |

> 📊 Gráfico comparativo: [`benchmark_chart.png`](./benchmark_chart.png)
> 📄 Métricas brutas: [`benchmark_report.json`](./benchmark_report.json)

### 3.3. Leitura crítica das métricas

O ganho de tempo é dramático (26×), mas o **pico de VRAM subiu 17 %** na versão otimizada — comportamento contraintuitivo que merece explicação:

- **No baseline (eager + sem cache)**: a cada novo token, Q/K/V de todos os 9.622 tokens prévios são **recalculados e descartados**. O pico de VRAM é dominado pelo forward pass instantâneo e por matrizes intermediárias temporárias, mas **nada permanece** entre tokens.
- **No otimizado (SDPA + KV cache)**: os tensores K e V de todos os 9.622 tokens são **mantidos em VRAM** entre passos (essa é a essência do cache). Esse acúmulo custa memória extra — cerca de 200 MB no nosso caso —, mas elimina o recálculo. É um **trade-off explícito de memória por tempo**.

O ganho conceitual do FA-2/SDPA está em **não materializar** a matriz de scores N×N (que para 9.622 tokens em FP16 ocuparia ~370 MB se materializada) — esse ganho é absorvido dentro dos kernels fundidos e por isso não aparece como redução no `max_memory_allocated`, embora seja o que viabiliza rodar sequências longas sem OOM. Para contextos maiores (32k, 128k tokens), o termo N² do baseline cresce quadraticamente e estouraria a VRAM da T4 — é nesse regime que a vantagem do SDPA/FA-2 fica explícita também no pico.

## 4. Parecer Técnico (Passo 5)

### 4.1. Parte A — Como QLoRA, KV Cache e FlashAttention salvaram o Transformer

O Transformer tradicional colapsa em produção por três custos compostos: o tamanho do checkpoint em FP16, o recálculo redundante de Q/K/V a cada token gerado e a materialização da matriz de atenção N×N na memória global da GPU. **QLoRA** atacou o primeiro problema reduzindo a pegada estática do modelo em ~3× ao quantizar os pesos para NF4 com double quantization — no nosso caso, o TinyLlama-1.1B ocupou apenas **735 MB** em vez dos ~2.200 MB teóricos em FP16. Sem isso, o checkpoint somado ao contexto de 9.622 tokens já comprometeria a margem de manobra da T4 (15 GB). **KV Cache** atacou o custo dinâmico da geração: a cada novo token, em vez de reprojetar Q, K e V de todo o prefixo, o modelo reutiliza os tensores K e V já computados, transformando uma complexidade total de O(L²·N) em O(L² + L·N). Como L (9.622) >> N (100), o ganho é dramático — observamos um **speedup de 26,19×** (180 s → 6,89 s) apenas com essa otimização de software. **FlashAttention-2** fecharia o ciclo no nível de hardware reformulando o cálculo do softmax(QKᵀ/√d) em *tiles* que cabem na SRAM on-chip da GPU em vez de materializar a matriz N×N na HBM, reduzindo a memória do passo de prompting de O(N²) para O(N). Como a T4 do Colab é Turing (sm_75) e o FA-2 oficial exige Ampere+, o notebook detectou a limitação e fez fallback para o **SDPA do PyTorch**, que aplica kernels memory-efficient equivalentes em espírito. As três técnicas são complementares: QLoRA economiza memória de pesos, KV Cache troca memória de cache por tempo de inferência, e FlashAttention-2/SDPA economiza memória de ativações — juntas, viabilizam executar em uma única GPU de consumidor o que antes exigiria um cluster.

### 4.2. Parte B — Por que com 2 milhões de tokens nem FlashAttention salva, e por que a indústria migraria para Mamba/SSM

Mesmo com FlashAttention-2 e KV Cache, o Transformer carrega uma cicatriz estrutural: o KV Cache cresce **linearmente** com o tamanho do contexto. Para uma janela de 2 milhões de tokens em um Llama-3 70B (8 KV heads, head_dim 128, 80 camadas, FP16), o cache sozinho ultrapassa **80 GB**, excedendo a VRAM de uma única H100 — e isso antes mesmo de considerar os pesos. FlashAttention reduz a constante e elimina a materialização da matriz de atenção, mas **não muda a complexidade assintótica** O(N) da memória do cache nem a O(N²) da fase de prompting (mantida na sub-camada de scores, ainda que tiled). A indústria respondeu com os **State Space Models** (Mamba, Mamba-2, RWKV) e arquiteturas híbridas (Jamba, Zamba), nas quais a "memória" do modelo é representada por um **estado oculto recorrente de tamanho fixo** atualizado por um operador seletivo linear — complexidade de memória **O(1)** por token gerado e tempo **O(N)** no contexto. Em outras palavras: enquanto o Transformer paga preço por cada token revisitado, o SSM compacta o passado em um vetor de estado constante. Para janelas de milhões de tokens (genoma, vídeo longo, prontuários médicos completos, código de monorepos inteiros), essa mudança de regime é o que torna a inferência viável em hardware comercial — e é por isso que pipelines como o nosso, hoje resolvidos com a tríade QLoRA + KV Cache + FlashAttention, tenderão a migrar para arquiteturas SSM ou híbridas no horizonte de produção.

## 5. Reflexões e Perguntas em Aberto

Durante a implementação, surgiram dúvidas que extrapolam o escopo do laboratório mas considerei valiosas para aprofundamento:

1. **FlashAttention-3 e Hopper** — O FA-3 explora *warp-specialization* assíncrona (WGMMA) exclusiva da H100. Para contextos abaixo de 128k tokens, o salto FA-2 → FA-3 entrega ganho real ou é dominado pela latência de I/O? Vale o esforço de portar pipelines de produção para Hopper ainda em 2026?

2. **Custo cognitivo do NF4 em domínio clínico** — A quantização 4-bit NF4 é praticamente "gratuita" em benchmarks gerais (MMLU, HellaSwag), mas em tarefas de raciocínio médico onde a precisão de dosagens e contraindicações importa, existe degradação mensurável? Estudos comparam NF4 vs INT8 vs FP16 em corpora clínicos como MedQA ou PubMedQA?

3. **Arquiteturas híbridas Mamba-Transformer** — Modelos como **Jamba** e **Zamba** intercalam blocos SSM e Attention para combinar recall preciso do segundo com complexidade O(1) do primeiro. Em um pipeline RAG médico, essa hibridez **substitui** a necessidade de retrieval externo (já que o SSM "lembra" mais barato) ou continua sendo complementar?

4. **PagedAttention (vLLM) vs KV Cache contíguo** — O PagedAttention do vLLM inspira-se em paginação de memória virtual para reduzir fragmentação em workloads multi-tenant. Em um cenário hospitalar com várias consultas simultâneas ao mesmo manual, o ganho prático sobre o KV Cache contíguo justifica a complexidade operacional adicional?

5. **Offloading de KV Cache para RAM/disco** — Para contextos extremos, algumas implementações (como o `OffloadedCache` do Hugging Face) movem KV cache antigo para a RAM ou NVMe. Que latência isso adiciona por token? Existe ponto de equilíbrio em que offloading + Transformer ainda supera um SSM puro?

## 6. Como executar

### 6.1. Pré-requisitos

- Conta Google e acesso ao [Google Colab](https://colab.research.google.com/)
- Runtime com GPU CUDA (ideal: L4 ou A100 para usar FA-2 nativo; T4 funciona com fallback SDPA)

### 6.2. Passos

1. Abrir o notebook no Colab via link direto:
   `https://colab.research.google.com/github/vini-haa/lab-10-pipeline-rag-qlora/blob/main/lab10_pipeline.ipynb`
2. Em `Ambiente de execução → Alterar o tipo de ambiente`, selecionar **GPU T4** (ou superior).
3. `Ambiente de execução → Executar tudo`. A célula de setup pode reiniciar a sessão automaticamente — basta clicar em **Executar tudo** uma segunda vez.
4. As métricas finais aparecem no console e são salvas em `benchmark_report.json` + `benchmark_chart.png`.

### 6.3. Estrutura do repositório

```
.
├── lab10_pipeline.ipynb        # Notebook principal (Passos 1–5)
├── benchmark_report.json       # Métricas brutas da execução
├── benchmark_chart.png         # Gráfico comparativo de tempo e VRAM
├── README.md                   # Este arquivo
└── .gitignore
```

## 7. Referências técnicas

- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.*
- Dao (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.*
- Gu & Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.*
- Hugging Face Transformers — documentação oficial de `BitsAndBytesConfig`, `attn_implementation` e `use_cache`.
