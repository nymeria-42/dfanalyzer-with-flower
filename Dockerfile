FROM ubuntu:22.04

# Install basic packages and Java 8
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    git \
    libcurl4-openssl-dev \
    psmisc \
    python3-pip \
    unzip \
    vim \
    wget \
    zip \
    && apt-get install -y openjdk-8-jdk

# Install Apache Maven 3.5.3
RUN wget https://repo.maven.apache.org/maven2/org/apache/maven/apache-maven/3.5.3/apache-maven-3.5.3-bin.tar.gz -P /tmp \
    && tar xf /tmp/apache-maven-*.tar.gz -C /opt

# Install MonetDB
RUN touch /etc/apt/sources.list.d/monetdb.list \
    && echo "deb [trusted=yes] https://dev.monetdb.org/downloads/deb/ jammy monetdb \
    \ndeb-src [trusted=yes] https://dev.monetdb.org/downloads/deb/ jammy monetdb" > /etc/apt/sources.list.d/monetdb.list \
    && wget --output-document=/etc/apt/trusted.gpg.d/monetdb.gpg https://dev.monetdb.org/downloads/MonetDB-GPG-KEY.gpg \
    && apt-get update && apt-get install -y monetdb5-sql monetdb-client

# Get DfAnalyzer repository
# RUN git clone https://gitlab.com/ssvitor/dataflow_analyzer.git && rm dataflow_analyzer/applications/dfanalyzer/dfa/backup/data-local.zip

# Configure environment variables
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64
ENV PATH $JAVA_HOME/bin:$PATH
ENV M2_HOME /opt/apache-maven-3.5.3
ENV MAVEN_HOME /opt/apache-maven-3.5.3
ENV PATH ${M2_HOME}/bin:${PATH}

# Install application's requirements
RUN pip install flwr tensorflow

# WORKDIR /dataflow_analyzer

# Copy files that provides fixes to usage of DfAnalyzer with the latest version of MonetDB and bigger upperbound limit to attribute text
COPY DfAnalyzer/ DfAnalyzer/
# COPY DfAnalyzer/data-local.zip DfAnalyzer/data-local.zip
# COPY DfAnalyzer/pom.xml DfAnalyzer/pom.xml
# COPY DfAnalyzer/DbConnection.java DfAnalyzer/src/main/java/rest/config/DbConnection.java
# COPY DfAnalyzer/DataflowProvenance.java DfAnalyzer/src/main/java/di/provenance/DataflowProvenance.java
# COPY DfAnalyzer/TaskProvenance.java DfAnalyzer/src/main/java/di/provenance/TaskProvenance.java
# COPY DfAnalyzer/WebConf.java DfAnalyzer/src/main/java/rest/server/WebConf.java
# COPY DfAnalyzer/start-dfanalyzer.sh DfAnalyzer/start-dfanalyzer.sh

# Prepare DfAnalyzer to be executed
RUN mvn -f DfAnalyzer/pom.xml clean package
# && cd maven && ./install_libraries.sh && cd .. \
# && mvn -f RawDataExtractor/pom.xml clean package \
# && mvn -f RawDataIndexer/pom.xml clean package \

RUN cd /DfAnalyzer/dfa-lib-python && make install

# Specify volumes to applications on container-side
VOLUME ["/applications/flowering"]
COPY requirements.txt .
RUN pip install -r requirements.txt
# Specify port to expose on container-side
EXPOSE 22000 50000 8080

CMD ["/bin/bash"]