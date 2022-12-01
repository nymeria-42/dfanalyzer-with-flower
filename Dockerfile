FROM ubuntu:22.04

# Install prerequisites

#instalando pacotes básicos
RUN apt-get update && apt-get install -y wget git libcurl4-openssl-dev psmisc zip unzip curl vim
RUN apt-get install -y build-essential

# instalando miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN . /opt/conda/etc/profile.d/conda.sh && conda init bash
RUN conda config --set auto_activate_base false

#instalando java 8
RUN apt-get install -y openjdk-8-jdk

#instalando apache maven 3.5.3
RUN wget https://repo.maven.apache.org/maven2/org/apache/maven/apache-maven/3.5.3/apache-maven-3.5.3-bin.tar.gz -P /tmp && tar xf /tmp/apache-maven-*.tar.gz -C /opt

#instalando monetdb
RUN touch /etc/apt/sources.list.d/monetdb.list && echo "deb [trusted=yes] https://dev.monetdb.org/downloads/deb/ jammy monetdb\ndeb-src [trusted=yes] https://dev.monetdb.org/downloads/deb/ jammy monetdb" > /etc/apt/sources.list.d/monetdb.list && wget --output-document=/etc/apt/trusted.gpg.d/monetdb.gpg https://dev.monetdb.org/downloads/MonetDB-GPG-KEY.gpg && apt-get update && apt-get install -y monetdb5-sql monetdb-client

# instalando o dfanalyzer
RUN git clone  https://gitlab.com/ssvitor/dataflow_analyzer.git && rm dataflow_analyzer/applications/dfanalyzer/dfa/backup/data-local.zip

# instalando fastbit - demora bastante
RUN git clone https://github.com/berkeleysdm/fastbit/ && cd fastbit && ./configure && make

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64
ENV PATH $JAVA_HOME/bin:$PATH
ENV M2_HOME /opt/apache-maven-3.5.3
ENV MAVEN_HOME /opt/apache-maven-3.5.3
ENV PATH ${M2_HOME}/bin:${PATH}

# RUN mvn -f dataflow_analyzer/DfAnalyzer/pom.xml clean package
# RUN cd /dataflow_analyzer/maven && ./install_libraries.sh
# RUN mvn -f dataflow_analyzer/RawDataExtractor/pom.xml clean package
# RUN mvn -f dataflow_analyzer/DfAnalyzer/pom.xml clean package
# RUN mvn -f dataflow_analyzer/RawDataIndexer/pom.xml clean package

COPY data-local.zip /dataflow_analyzer/applications/dfanalyzer/dfa/backup
COPY data-local.zip /dataflow_analyzer/DfAnalyzer/data-local.zip
COPY flower-studies /dataflow_analyzer/applications/flower-studies
COPY pom.xml /dataflow_analyzer/DfAnalyzer/pom.xml

EXPOSE 22000

CMD ["/bin/bash"]
