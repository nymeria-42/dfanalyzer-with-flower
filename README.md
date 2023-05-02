# Flower-DfAnalyzer

A partir da pasta contendo Dockerfile e arquivos a serem copiados para o container, constuir a imagem rodando:

```bash
sudo service docker start
docker build -t dfanalyzer .
```

Para criar dataset_partitions
```bash
git clone git@github.com:alan-lira/dataset-splitter.git -b develop
```

Para iniciar o DfAnalyzer:

```bash
docker compose up dfanalyzer
```

- A pasta com as partições criada é usada como volume aqui para inserção no MonetDB

Para carregar dados do particionamento no MonetDB:

```bash
cd dataset_partitions_monetdb && virtualenv venv && . venv/bin/activate && pip install -r requirements.txt && python import_monetdb.py
```

Para iniciar servidor do MongoDB:

```bash
docker compose up mongodb
```

Para rodar a proveniência prospectiva:

```bash
docker compose up prospective-provenance
```

Para iniciar o servidor do flower:

```bash
docker compose up server
```

Para iniciar um cliente:

```bash
docker compose up client1
```

Para iniciar 5 clientes com o dataset já particionado:

```bash
docker compose up client1 client2 client3 client4 client5
```

É possível, então, consultar o banco MonetDB:

```bash
docker exec -it dfanalyzer mclient -u monetdb -d dataflow_analyzer
#password: monetdb
```

Para listar as tabelas que não são do sistema:

```sql
SELECT tables.name FROM tables WHERE tables.system=false ;
```

Para ver a interface web do DfAnalyzer, acessar [`http://localhost:22000`](http://localhost:22000/)

- É possível visualizar o grafo com as tasks do workflow executado.
- OBS: queries pela interface gráfica não estão funcionais.

Link do repositório do DfAnalyzer: [`https://gitlab.com/ssvitor/dataflow_analyzer`](https://gitlab.com/ssvitor/dataflow_analyzer)
