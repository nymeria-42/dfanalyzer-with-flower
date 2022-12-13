# DfAnalyzer

A partir da pasta contendo Dockerfile e arquivos a serem copiados para o container:

```bash
sudo service docker start
sudo docker build -t dfanalyzer .
sudo docker run -it -p 22000:22000 --name dfanalyzer_container dfanalyzer bash 
```

Para iniciar o DfAnalyzer:

```bash
cd dataflow_analyzer/Dfanalyzer
./start-dfanalyzer.sh
```

Para acessar o container criado:

```bash
sudo docker exec -it dfanalyzer_container bash
```

Para iniciar o servidor do flower, rodar em outro terminal:

```bash
cd dataflow_analyzer/applications/flower-studies
conda create -n flower-studies
conda activate flower-studies
conda install pip
pip install flwr tensorflow
make init

python3 dfanalyzer-tf_cifar/server.py
```

Para iniciar um cliente, rodar em outro terminal:

```bash
cd dataflow_analyzer/applications/flower-studies/
conda activate flower-studies
python3 tf_cifar/client.py
```

É possível, então, consultar o banco MonetDB, rodando em outro terminal:

```bash
mclient -u monetdb -dataflow_analyzer
#password: monetdb
```

Para listar as tabelas que não são do sistema:

```sql
SELECT tables.name FROM tables WHERE tables.system=false ;
```

Para ver a interface web do DfAnalyzer, acessar [`http://localhost:22000`](http://localhost:22000/)

- É possível visualizar o grafo com as tasks do workflow executado.
- OBS: queries pela interface gráfica não estão funcionando.

Link do repositório do DfAnalyzer: [`https://gitlab.com/ssvitor/dataflow_analyzer`](https://gitlab.com/ssvitor/dataflow_analyzer)
