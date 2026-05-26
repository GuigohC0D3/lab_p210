# Laboratório 10: O Pipeline Definitivo (RAG, QLoRA e Otimização de Inferência na GPU)

> **Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por Guilherme Pereira**

---

## Descrição

Pipeline de IA ponta a ponta simulando um ambiente de produção onde:
1. Um **RAG** recupera ~12.000 tokens de manuais médicos fictícios
2. Um modelo **Llama** quantizado em **4-bits (QLoRA)** processa o contexto
3. A geração é otimizada com **KV Cache** + **FlashAttention-2** para evitar OOM na GPU

---

## Métricas de Benchmark

> Valores de referência obtidos em GPU NVIDIA (resultados exatos disponíveis após execução do notebook)

### Passo 1 — Carregamento com QLoRA 4-bits

| Métrica | Valor |
|---|---|
| Modelo | TinyLlama-1.1B-Chat-v1.0 |
| Precisão dos pesos | 4-bit NF4 |
| Compute dtype | float16 |
| **VRAM do modelo (4-bit)** | **~550–650 MB** |
| VRAM estimada em float16 | ~2.200 MB |
| **Redução de memória** | **~75%** |

### Passo 3 vs Passo 4 — Geração de 100 tokens

| Configuração | Tempo (s) | Velocidade (tok/s) | Pico VRAM (MB) |
|---|---|---|---|
| **Sem KV Cache** (baseline) | ~45–90s | ~1–2 | ~4.000–6.000 |
| **KV Cache + FlashAttention-2** | ~3–8s | ~15–30 | ~1.200–1.800 |
| **Ganho** | **~10–15x mais rápido** | — | **~70% menos VRAM** |

> Execute `lab10.ipynb` para obter os valores exatos na sua GPU.

---

## Parecer Técnico Arquitetural

### Parte A: Como QLoRA + KV Cache + FlashAttention-2 salvaram o pipeline do colapso de VRAM

O principal gargalo de memória em um Transformer padrão opera em três frentes simultâneas: os próprios pesos do modelo, as matrizes de atenção intermediárias e o estado gerado a cada novo token. Neste laboratório, cada frente foi atacada por uma técnica específica. O **QLoRA em 4-bits** resolve o custo estático dos pesos: ao representar cada parâmetro em NormalFloat4 em vez de float16, o modelo de 1.1B de parâmetros cabe em ~600 MB de VRAM em vez dos ~2.200 MB originais, liberando memória para o contexto massivo do RAG. O **KV Cache** elimina o custo dinâmico da fase de decodificação: sem ele, a cada novo token gerado o modelo recalcula as matrizes de Keys e Values para todos os tokens anteriores, criando uma complexidade O(n) de custo por passo e O(n²) acumulado nos 100 passos — exatamente o recálculo redundante que travou o servidor. Com o cache ativo, os vetores K e V da fase de *prefilling* (os 12.000 tokens do RAG) são calculados uma única vez e reutilizados, tornando cada passo de decodificação O(1) em relação ao contexto. Por fim, o **FlashAttention-2** ataca o gargalo de hardware: o Attention padrão materializa a matriz de atenção n×n na DRAM da GPU (memória HBM, lenta), consumindo O(n²) de memória. O FlashAttention-2 reorganiza o cálculo em blocos que cabem inteiramente na SRAM da GPU (cache L1, ~100x mais rápida), nunca escrevendo a matriz completa — resultado: O(n) de memória e throughput dramaticamente superior. A combinação das três técnicas permitiu executar um contexto de ~12.000 tokens em hardware de consumo sem OOM.

### Parte B: Por que FlashAttention falha a 2 milhões de tokens e por que a indústria precisa dos State Space Models

O FlashAttention resolve o problema de *bandwidth* (transferência de dados entre HBM e SRAM) e reduz a pegada de memória para O(n) em vez de O(n²) — mas **não elimina a dependência quadrática de computação**: o número de operações de ponto flutuante ainda escala como O(n²) em relação ao tamanho da sequência. Com 2 milhões de tokens, mesmo que a memória fosse suficiente, o número de FLOPs se tornaria astronomicamente grande: a proporção entre 2.000.000 e 15.000 tokens é de ~133x, e em complexidade quadrática isso representa ~17.700x mais operações de atenção por camada. Em paralelo, o **KV Cache** a 2M tokens exigiria armazenar 2 milhões de pares K,V por camada por cabeça de atenção — em modelos com 32 camadas e 32 cabeças, isso ultrapassa facilmente 50–100 GB de VRAM só para o cache. Por essas razões, a indústria está migrando para arquiteturas como **Mamba (State Space Models — SSMs)**: em vez de calcular a atenção entre todos os pares de tokens, os SSMs comprimem toda a informação da sequência passada em um **estado oculto de tamanho fixo** (uma matriz de estado), que é atualizada de forma recorrente a cada novo token. Isso garante complexidade de memória **O(1)** em relação ao comprimento da sequência — o estado tem sempre o mesmo tamanho, independentemente de serem 15.000 ou 2.000.000 tokens processados — e complexidade de computação **O(n)** por passo de geração, tornando o processamento de contextos de escala de livros e repositórios inteiros computacionalmente viável em hardware real.

---

## Como Executar

```bash
# 1. Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Instalar FlashAttention-2 (requer Long Path habilitado no Windows — ver abaixo)
pip install flash-attn --no-build-isolation

# 3. Executar o pipeline
python -m src.main
```

**Requisitos de hardware:**
- GPU NVIDIA com suporte a CUDA (mínimo 8 GB VRAM recomendado)
- FlashAttention-2 requer GPU com compute capability ≥ 8.0 (Ampere ou superior: RTX 3000+, A100, H100)

**Windows — Long Path (obrigatório para flash-attn):**

O `flash-attn` compila do código-fonte e usa caminhos de arquivo maiores que 260 caracteres.
Execute o comando abaixo no PowerShell **como Administrador** e reinicie o terminal:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

---

## Estrutura do Repositório

```
lab_p210/
├── src/
│   ├── config.py        # MODEL_ID e constantes
│   ├── utils.py         # helpers de VRAM e CUDA
│   ├── rag.py           # geração do contexto RAG (~12k tokens)
│   ├── model.py         # carregamento QLoRA 4-bit + FlashAttention-2
│   ├── benchmark.py     # geração com/sem KV Cache + métricas
│   ├── visualize.py     # relatório comparativo + gráfico
│   └── main.py          # orquestrador do pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Referências

- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023.
- Dao et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS 2022.
- Dao (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. ICLR 2024.
- Gu & Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv 2312.00752.
