import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



# Carregar o arquivo CSV para inspeção
file_path = r"C:\Users\hanie\Downloads\estudo.csv"

df = pd.read_csv(file_path)

print(df)
# Exibir as primeiras linhas para entender os dados
df.head(), df.columns, df.info()


#--------ANÁLISE EXPLORATÓRIA DE DADOS--------

# Carregar os dados
df = pd.read_csv('C:/Users/hanie/Downloads/estudo.csv')

# Visualizar as primeiras linhas
print(df.head())

# Estatísticas descritivas
print(df.describe())

# Visualização de distribuição de uma coluna 
sns.histplot(df['product'], bins=30) #alterar para coluna que gostaria de ver (product está apenas como exemplo)
plt.show()

# Matriz de correlação
plt.figure(figsize=(10, 8))
#sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()


#--------MODELAGEM E PREVISÃO--------

# Separar as variáveis independentes e dependentes
X = df[['quantity']]  
y = df['price_y']  # Coluna que quer prever

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar e treinar o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')



#--------SEGMENTAÇÃO E AGRUPAMENTO--------

# Normalizar os dados (opcional, dependendo dos dados)
scaler = StandardScaler()
# Selecione as colunas que você deseja normalizar
X = df[['price_y', 'price_x', 'quantity']]
X_scaled = scaler.fit_transform(X)

# Aplicar K-Means
kmeans = KMeans(n_clusters=3)  # Ajuste o número de clusters conforme necessário
kmeans.fit(X_scaled)

# Adicionar os rótulos de cluster aos dados
df['Cluster'] = kmeans.labels_

# Criar os gráficos de dispersão
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Combinação 1: 'price_y' vs 'quantity'
sns.scatterplot(x='price_y', y='quantity', hue='Cluster', data=df, ax=axes[0])
axes[0].set_title('price_y vs quantity')

# Combinação 2: 'price_x' vs 'quantity'
sns.scatterplot(x='price_x', y='quantity', hue='Cluster', data=df, ax=axes[1])
axes[1].set_title('price_x vs quantity')

# Combinação 3: 'price_y' vs 'price_x'
sns.scatterplot(x='price_y', y='price_x', hue='Cluster', data=df, ax=axes[2])
axes[2].set_title('price_y vs price_x')

# Ajustar layout e mostrar os gráficos
plt.tight_layout()
plt.show()

df.to_csv('dados_com_clusters.csv', index=False)
