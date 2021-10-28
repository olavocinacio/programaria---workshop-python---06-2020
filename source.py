import panda as pd #permite o trabalho com dataframes
import matplotlib as plt #plotar gráficos
import seaborn as sbn #plotar gráficos
from plotnine import * #também para plotar gráficos
from sklearn.model_selection import trains_test_split7 #DIvisão do dataset entre treinamento e teste
from sklearn.preprocessing import StandardScale #Criação de um gráfico de calor para relação entre variáveis
from sklearn.linear_model import LogisticRegression #Regressão logística
from sklearn import metrics  #Criação da matriz de confusão
import pickle #Será utilizado para salvar o modelo criado

df = pd.read_csv('nome.csv'); #abrindo o dataset
df.head(); #mostra as 5 primeiras linhas. Importante para checar o cabelçalho dos dados
df.shape; #retorna (Linhas, colunas)
df.dypes; #retorna as variáveis e o tipo das mesmas
list(df.columns); #retorna uma lista com o nome das colunas
#Analisar quais variávei provocam alterações no resultado buscado e eliminar as indiferentes
df.drop('coluna', axis = 1, inplace = True); #elimina a coluna
df['coluna'].value_conts() #Mostra uma relação entre um dado e o número de vezes que ele aparece
df.isna().any(); #Se True, significa que existe dados faltantes na coluna relacionada
df.insna().sum(); #Mostra o número de linhas vazias em cada coluna
df.drpna(axis = 0, subset['variável1', 'variável2'], inplace = True); #Remove todas as linhas das variáveis colocadas
df['coluna'].fillna('substituto', inplace = True); #substitui os faltantes pelo substituto 
df['coluna'].median; #calcula a mediana dos dados 
#É importante não deixar nenhuma célula vazia
df.duplicated().sum(); #Checa entradas duplicadas(dados repetidos)
df.drop_duplicates(inplace = True); #Retira as duplciatas
df['nova_coluna'] = df['coluna1'].sub(df['coluna2'], axis = 0) #Cria uma nova coluna a partir da diferença entre a coluna2 e a coluna1
df['nova_coluna2'] = (df['coluna1']/df['coluna2']*100) #Cria uma nova coluna que representa a porcentagem da coluna1 em relação à coluna2
#Os exemplos acima servem para exemplificar o processo de criação de uma nova coluna a partir da operação de colunas já existentes
df_to_csv('nove.csv', index = False); #Salva o dataframe como um novo arquivo csv
ggplot() #
#Plotar os gráficos é algo interessante para analisar a relação entre as variáveis
#Variáveis dependentes não são interessantes para o treinamento de um modelo
df.corr() #Pode ser utilizado para mostrar um índice de relação entre as variáveis em um mapa de calor. Quanto mais claro, mais forte a relação
#Deve-se remover as variáveis totalmente relacionadas com as outras, podendo ser substituídas por variáveis menos agravantes
#Uma taxa aceitável está abaixo de 0.65
#É mais fácil prever um grupo a um número exato.Portanto, é interessante gerar classificações ao trabalhar com a previsão numérica
df = pd.get_dummies(data = df, columns = ['coluna'], prefix['noovo_nome'], drop_first = True); #Transforma as informçaões de um dado classificatório em uma matriz com todas as opções, marcando 1(True) no correspondente e 0(False) nas outras classificações 
#Para o aprendizado supervisionado, é importante retirar a coluna com as "respostas"(dado que deseja-se estimar) dos dados que serão utilizados na predição. 
X = pd.DataFrame(columns = ['coluna1','coluna2'], data = df);
#A coluna que contém a variável desejada será utilizada para que o modelo cheque a acertabilidade dele mesmo.
y = pd.DataFrame(columns = ['coluna3'], data = df); #Nesse exemplo, serão utilizadas as colunas 1 e 2 para prever o resultado da coluna
#Modelo tenta estimar um valor a partir das variáveis dadas -> Checa com o resultado real -> Se ajusta automaticamente e reinicia o teste
#Divisão entre treinamento, teste e validação.  
X_train, X_test, y_train, y_test = trains_test_split(X, y, test_size = 0.3, random_state = 42) #Geralmente 80% treinamento, 10% teste, 10% validação. Nesse caso será 70% para treinamento e 30% para teste
#Normalizar os dados -> Ter uma amplitude muito grande entre os dados é ruim para o cálculo do erro do modelo. Portanto faz-se uma escala de 0 a 1 para os memsos.
sc_X = StandardScaler(); #Inicialização do construtor
X_train = sc_X.fit_transform(X_train); #Aplica normalizador no conjunto de treinamento
X_test = sc_X.transform(X_test); #Aplicação do normalizador no conjunto de testes
#Modelo de regressão logística -> Disponível no sklearn
logit = LogisticRegression(verbose = 1, max_iter = 1000); #verbose -> Para que se torne possível a análise simultânea ao que está acontecendo. max_iter -> Número de iteração máximas
logit.fit(X_train, np.ravel(y_train, order = 'C')); #Serve para realimentar o modelo com os dados gerados
y_pred = logit.predict(X_test); #Tentativa de predição dos dados de teste. Vai retornar uma lista com os dados previstos
cnf_matrix = metrics.confusion_matrix(y_test, y_pred);
plot_confusion_matrix(cnf_matrix, calsses = ['a', 'b', 'c', 'd'], title = 'Matriz de confusão não normalizada', normalize = False); #Função criada para visulização da matriz de confusão -> Necessário copiar do código fonte da palestrante
#Matriz de confusão -> Y = Classe real e X = Classe que o modelo fez a previsão
#É importante tratar as classes desbalanceadas para ter uma melhor predição
metrics.classification_report(y_test, y_pred, target_names = ['a','b','c','d']); #retorna um resumo do modelo treinado
#Salvando o modelo:
modelo_treinado = 'nome_do_modelo.sav';
pickle.dump(logit, open(modelo_treinado), 'wb');
#Carregando um modelo:
modelo_carregado = pickle.load(open(modelo_treinado, 'rb'));
#Mostrando um novo dado ao modelo:
novo_dado = ['dado_coluna1','dado_coluna2'];
modelo_carregado.predict([novo_dado]);


