# 🔁 Validação cruzada e split estratificado

[⬅️ README](../README.md) · [Dados](dados-sinteticos.md) · [German Credit](german-credit.md) · [Modelo](modelo.md) · [Métricas](metricas.md) · **Validação cruzada** · [Mitigação](mitigacao.md) · [Inferência](inferencia.md) · [Serviço](servico.md) · [API](api.md) · [Testes](testes.md)

---

## 🔁 Validação cruzada e split estratificado

Todo número deste projeto até aqui saiu de **uma** divisão: 800 clientes para treinar, 200 para medir. Isso tem dois problemas, e nenhum deles é o modelo.

**A régua se mexe.** Um corte cru depois do embaralhamento acerta a proporção de inadimplentes *em média* e erra em qualquer execução específica. Com 30% de positivos e 200 linhas de teste, o sorteio entregava entre `0.6500` e `0.7600` de [baseline da classe majoritária](metricas.md#-matriz-de-confusão) — e é contra esse número que toda acurácia deste projeto é lida. Onze pontos de oscilação na régua entram direto na conclusão.

**A incerteza não aparece.** "Acurácia `0.7088`" parece uma medida. Não é: é uma amostra de tamanho 1 de uma distribuição cuja largura ninguém viu.

As duas coisas têm conserto, e são independentes.

### Estratificar: parar de sortear a régua

Estratificar é preservar a proporção de classes nos dois lados do corte:

```javascript
const stratifiedSplitCustomers = (customers, trainRatio = 0.8) => {
  const training = new Set();

  [...new Set(customers.map(({ risk }) => risk))].forEach((label) => {
    const indexes = customers
      .map((customer, index) => ({ risk: customer.risk, index }))
      .filter((row) => row.risk === label);

    indexes
      .slice(0, Math.floor(indexes.length * trainRatio))
      .forEach(({ index }) => training.add(index));
  });

  return {
    trainCustomers: customers.filter((ignored, index) => training.has(index)),
    testCustomers: customers.filter((ignored, index) => !training.has(index)),
  };
};
```

O efeito é imediato e exato. Nas mesmas 15 divisões que antes davam de `0.6500` a `0.7600`:

| Baseline do conjunto de teste | Corte cru | Estratificado |
| --- | ---: | ---: |
| German Credit, 15 divisões | `0.7027` ± 0.0080, variando `0.6500`–`0.7600` | **`0.7000` em todas as 15** |
| Sintético, 15 divisões | `0.8431` ± 0.0078, variando `0.7833`–`0.9000` | **`0.8423` em todas as 15** |

O `0.7000` não é média: é o valor de **cada** divisão. A régua parou de se mexer, e a variação que sobra na acurácia passa a ser do modelo, não do sorteio.

> ⚠️ **Um detalhe de implementação que é um bug esperando para acontecer.** A função devolve os clientes na ordem em que eles chegaram, não agrupados por rótulo. O motivo: `fit` reserva os **últimos 20%** do treino para validação **antes** de embaralhar. Um conjunto ordenado por classe daria uma fatia de validação quase toda de um rótulo só — e o `val_loss` que governa o *early stopping* estaria medindo outra coisa. Há um teste que trava a ordem.

E o que a estratificação **não** faz: ela preserva a proporção de *inadimplentes*, não a de *mulheres*. As taxas-base por sexo continuam variando entre divisões (`35,2%` ± 1,3 contra `27,7%` ± 0,5), e estratificar pelo atributo protegido seria outra decisão — com outras implicações, porque significaria usar a coluna que o modelo não recebe para montar o experimento.

### Validar cruzado: k estimativas em vez de uma

A validação cruzada divide o dataset em k dobras, treina k vezes, e cada cliente é pontuado **exatamente uma vez** por um modelo que não o viu treinar. As dobras também são estratificadas, então todas as k medidas usam a mesma régua.

```bash
npm run cv                       # 5 dobras no German Credit
node index.js --cv=10            # 10 dobras
node index.js --source=synthetic --cv
```

```text
Fonte: German Credit — UCI/Statlog (Hofmann, 1994), one-hot
Validação cruzada estratificada: 5 dobras

Dobra | Treino | Teste | Baseline | Acurácia |      AUC | Limiar | Custo
------+--------+-------+----------+----------+----------+--------+------
1     |    800 |   200 |   0.7000 |   0.7400 |   0.7855 | 0.1777 |    92
2     |    800 |   200 |   0.7000 |   0.6900 |   0.7065 | 0.0983 |   113
3     |    800 |   200 |   0.7000 |   0.7800 |   0.7954 | 0.1560 |   105
4     |    800 |   200 |   0.7000 |   0.7300 |   0.7779 | 0.2100 |    95
5     |    800 |   200 |   0.7000 |   0.7800 |   0.8230 | 0.2261 |    84
Média |        |       |   0.7000 |   0.7440 |   0.7776 |        |  97.8
Erro  |        |       | ± 0.0000 | ± 0.0169 | ± 0.0193 |        | ± 5.1

AUC sobre o dataset inteiro (curva única, score fora da amostra): 0.7714
```

A dobra 2 acerta `0.6900` — **abaixo do baseline**. A dobra 5 acerta `0.7800`. São o mesmo código, o mesmo dataset e a mesma configuração; a diferença de nove pontos é inteiramente o sorteio de quais 200 clientes ficaram de fora. Reportar qualquer uma delas sozinha seria reportar ruído com quatro casas decimais.

Dois detalhes que não são decorativos:

- **O scaler é remedido em cada dobra.** Reaproveitar uma escala ajustada no dataset inteiro vazaria o teste de cada dobra para dentro do treino — e o erro seria invisível, porque o número sairia *melhor*.
- **A AUC do dataset inteiro não é a média das AUCs.** `0.7714` vem de uma curva única, montada com os 1.000 scores fora da amostra; `0.7776` ± 0.0193 é a média de cinco curvas independentes. As duas respondem perguntas diferentes e não têm por que coincidir.

### O que isso mudou na conclusão do projeto

Repetindo a validação cruzada **10 vezes**, com sementes diferentes — 50 treinos:

| | Divisão fixa (semente `42`) | Validação cruzada |
| --- | ---: | ---: |
| Baseline | `0.7000` | `0.7000` |
| Acurácia | `0.7088` ± 0.0026 | **`0.7499`** ± 0.0018 |
| **Ganho sobre o baseline** | **+0,9 pt** | **+5,0 pts** |
| AUC | `0.7387` ± 0.0013 | `0.7807` ± 0.0014 |
| Custo `5:1` no limiar escolhido | `105.4` ± 0.7 | `96.7` ± 0.6 |

**A semente `42` é uma divisão ruim** — e não por pouco: nas 15 divisões sorteadas para conferir, a pior deu `0.7150`, ainda acima dela. Duas medições independentes concordam entre si e discordam dela: a média de 15 divisões dá `0.7467` ± 0.0058 e a validação cruzada dá `0.7499` ± 0.0018.

Isso **muda uma conclusão que este projeto já publicou**, na [comparação lado a lado](german-credit.md#comparação-lado-a-lado). Com a divisão fixa e o corte cru, o modelo empatava com o baseline, e o texto dizia que o ganho era zero. Com a régua estabilizada e a estimativa feita como se deve, o ganho existe: **+5,0 pontos**, ou seja, cerca de 50 clientes a mais classificados corretamente em cada 1.000. O que continua verdadeiro é a lição maior — a acurácia sozinha não diz isso, e foram a [AUC](metricas.md#-curva-roc-e-auc) e o [ajuste do limiar](metricas.md#-ajuste-do-limiar-de-decisão) que mostraram onde estava o problema.

> 🔬 A semente continua fixa no código. Sabendo que existem divisões que dão `0.7900`, a tentação de procurá-las é real, e uma semente congelada é a defesa mais barata contra escolher o resultado depois de ver os resultados. O que mudou é que agora existe um comando — `npm run cv` — que responde a pergunta certa sem depender dela.

### O que ela permitiu medir

O ganho maior não foi na acurácia: foi na [auditoria de disparidade](german-credit.md#a-coluna-que-o-modelo-não-recebeu). Com um hold-out, ela roda sobre cerca de **66 mulheres** e a razão de aprovação balança de `0.71` a `1.33` conforme o sorteio. Com validação cruzada, **todos os 1.000 clientes** têm um score fora da amostra, e a mesma auditoria roda sobre **310 mulheres** de uma vez.

É essa a diferença entre relatar um número e medir um efeito — e é o que torna a [mitigação](mitigacao.md#-mitigação-da-disparidade) verificável em vez de plausível.

**O preço:** k treinos em vez de um. Para 1.000 linhas e uma rede de 1.073 parâmetros isso é meio minuto; é por isso que a validação cruzada é um **comando separado**, e não o caminho padrão do `npm start` — que continua servindo para ver o pipeline inteiro rodar, com matriz, curva e limiar, em uma passada só.

---

[⬅️ Voltar ao README](../README.md)
