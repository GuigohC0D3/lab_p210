from transformers import AutoTokenizer

CAPITULO_TEMPLATE = """
Capítulo {n}: {titulo}

1. Fisiopatologia
A fisiopatologia desta condição envolve uma cascata inflamatória mediada por citocinas pró-inflamatórias,
incluindo TNF-α, IL-6 e IL-1β. A ativação do complemento desempenha papel central na progressão da doença,
com consequente disfunção endotelial e aumento da permeabilidade vascular. Estudos de coorte prospectivos
demonstraram correlação positiva entre marcadores séricos de inflamação (PCR, VHS, ferritina) e desfechos
clínicos adversos. A disfunção mitocondrial observada em biópsias teciduais sugere comprometimento da
fosforilação oxidativa, resultando em estresse oxidativo celular com produção exacerbada de espécies reativas
de oxigênio (ROS). O comprometimento da barreira hematoencefálica, quando presente, correlaciona-se com
manifestações neurológicas incluindo encefalopatia metabólica, convulsões e alterações cognitivas.

2. Diagnóstico Diferencial
O diagnóstico diferencial deve considerar etiologias infecciosas, autoimunes, neoplásicas e metabólicas.
A investigação laboratorial de primeira linha inclui hemograma completo com diferencial, painel metabólico
completo, função hepática e renal, coagulograma, marcadores inflamatórios e sorologias direcionadas.
Exames de imagem como tomografia computadorizada com contraste e ressonância magnética são indicados
quando há suspeita de acometimento de órgãos específicos. A biopsia tecidual com análise histopatológica
e imunohistoquímica permanece o padrão ouro para confirmação diagnóstica em casos selecionados.
A análise citogenética e molecular é essencial no contexto de doenças hematológicas e neoplasias sólidas.

3. Protocolo Terapêutico
O manejo terapêutico baseia-se em evidências de ensaios clínicos randomizados (ECRs) de alta qualidade.
A terapia de primeira linha consiste em corticosteroides sistêmicos (prednisona 1 mg/kg/dia) com redução
gradual conforme resposta clínica e laboratorial. Imunossupressores como azatioprina, micofenolato mofetil
e ciclofosfamida são reservados para casos refratários ou com contraindicação aos corticosteroides.
Agentes biológicos direcionados, incluindo inibidores de TNF-α (infliximabe, adalimumabe, etanercepte),
inibidores de IL-6 (tocilizumabe, sarilumabe) e inibidores de JAK (baricitinibe, tofacitinibe), representam
avanços terapêuticos significativos para subgrupos específicos de pacientes.

4. Monitoramento e Desfechos
O monitoramento da resposta terapêutica deve ser realizado com avaliações clínicas seriadas e reavaliação
laboratorial a cada 4-8 semanas. Critérios de remissão incluem ausência de atividade clínica, normalização
de marcadores inflamatórios e estabilização de parâmetros funcionais. Desfechos adversos incluem falência
orgânica, infecções oportunistas secundárias à imunossupressão e toxicidade medicamentosa. A qualidade
de vida relacionada à saúde (QVRS) deve ser avaliada por instrumentos validados como SF-36, EQ-5D e
escalas específicas da doença. O seguimento de longo prazo é essencial para detecção precoce de recidivas.

5. Considerações Especiais em Populações Vulneráveis
Populações especiais como gestantes, idosos, crianças e pacientes imunossuprimidos requerem adaptações
no protocolo diagnóstico e terapêutico. Em gestantes, a teratogenicidade dos medicamentos deve ser
cuidadosamente avaliada, com preferência por agentes com maior segurança estabelecida no perfil
materno-fetal. Em idosos, a polifarmácia, as interações medicamentosas e a redução da reserva renal
e hepática exigem ajustes de dose e monitoramento mais frequente. Em pacientes pediátricos, doses
baseadas em peso corporal e considerações sobre o impacto no crescimento e desenvolvimento são
fundamentais para o planejamento terapêutico individualizado e centrado no paciente.
"""

TITULOS = [
    "Síndrome Inflamatória Multissistêmica — Diagnóstico e Manejo",
    "Doenças Autoimunes Sistêmicas — Classificação e Tratamento",
    "Insuficiência Cardíaca Congestiva — Fisiopatologia e Terapêutica",
    "Sepse e Choque Séptico — Protocolo de Ressuscitação",
    "Neoplasias Hematológicas — Diagnóstico Molecular e Imunoterapia",
]


def build_rag_context(tokenizer: AutoTokenizer) -> tuple[str, int]:
    capitulos = [CAPITULO_TEMPLATE.format(n=i + 1, titulo=TITULOS[i]) * 4 for i in range(5)]
    contexto = "\n\n".join(capitulos)
    n_tokens = len(tokenizer.encode(contexto))
    print(f"=== MÉTRICA PASSO 2 ===")
    print(f"Tokens no contexto RAG: {n_tokens:,}")
    print(f"Caracteres no texto: {len(contexto):,}")
    return contexto, n_tokens
