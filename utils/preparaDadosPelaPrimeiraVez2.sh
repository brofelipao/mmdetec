# Remove resultados da última execução
# Remove a categoria a mais que havia no banco de imagens dos ovos de aedes
# Troca ovos e ovo por Corn (para não ter que mexer dentro do código principal)
# Faz a divisão das anotações em dobras para a validação cruzada

./apagaResultados.sh
./removeCategoriaZero.sh
./trocaPorCorn.sh
python geraDobras.py -folds=5 -valperc=0.3
