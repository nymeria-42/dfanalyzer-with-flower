# Flower-DfAnalyzer

A partir da pasta contendo Dockerfile e arquivos a serem copiados para o container, constuir a imagem rodando:

```bash
sudo service docker start
sudo docker build -t dfanalyzer .
```

Para iniciar o DfAnalyzer:

```bash
sudo docker-compose up dfanalyzer
```

Para iniciar o servidor do flower:

```bash
sudo docker-compose up server
```

Para iniciar um cliente:

```bash
sudo docker-compose up client
```

É possível, então, consultar o banco MonetDB, acessando o container rodando o DfAnalyzer em outro terminal:

```bash
sudo docker exec -it dfanalyzer

mclient -u monetdb -d dataflow_analyzer
#password: monetdb
```

Para listar as tabelas que não são do sistema:

```sql
SELECT tables.name FROM tables WHERE tables.system=false ;
```

Para ver a interface web do DfAnalyzer, acessar [`http://localhost:22000`](http://localhost:22000/)

- É possível visualizar o grafo com as tasks do workflow executado.
- OBS: queries pela interface gráfica não estão funcionais.

Para ver o endereço IP do container do dfanalyzer:
```docker
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' dfanalyzer
```
Link do repositório do DfAnalyzer: [`https://gitlab.com/ssvitor/dataflow_analyzer`](https://gitlab.com/ssvitor/dataflow_analyzer)
