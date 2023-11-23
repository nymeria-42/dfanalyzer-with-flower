from collections import defaultdict
import time

from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path

from pymonetdb import connect

def get_connection_monetdb(monetdb_settings):

    connection = connect(
        hostname=monetdb_settings["hostname"],
        port=monetdb_settings["port"],
        username=monetdb_settings["username"],
        password=monetdb_settings["password"],
        database=monetdb_settings["database"],
    )

    return connection

def query_monetdb(conn, query, _all=False, only_first=True):
    cursor = conn.cursor()
    cursor.execute(operation=query)

    if _all:
        if only_first:
            return [item[0] for item in cursor.fetchall()]
        return cursor.fetchall()
    else:
        result = cursor.fetchone()
        if result and only_first:
            return result[0]
        return result

def main():
    ag = ArgumentParser()
    ag.add_argument(
        "--server_config_file",
        type=Path,
        required=True,
    )
    parsed_args = ag.parse_args()
    server_config_file = Path(parsed_args.server_config_file)

    cp = ConfigParser()
    cp.optionxform = str
    cp.read(filenames=server_config_file, encoding="utf-8")
    monetdb_settings = cp["MonetDB Settings"]

    fails = defaultdict(int)

    _server_round = 0
    _experiment_id = 0

    conn = get_connection_monetdb(monetdb_settings)
    query = f"""
        SELECT num_rounds FROM iserverconfig WHERE experiment_id = {_experiment_id};
    """
    num_rounds = int(query_monetdb(conn, query))

    while _server_round != num_rounds:

        conn = get_connection_monetdb(monetdb_settings)
        
        query = f"""
        SELECT client_id FROM idatasetload WHERE experiment_id = {_experiment_id};
        """
        
        # lista com ids de clientes que já entraram no treinamento
        list_ids = query_monetdb(conn, query, _all=True)
        
        clients_ids = set(list_ids)
        
        query = f"""
            SELECT server_round FROM oservertrainingaggregation
            WHERE experiment_id = {_experiment_id}
            ORDER BY server_round DESC
            LIMIT 1;
        """
        _server_round = int(query_monetdb(conn, query))
        
        for _client_id in clients_ids:
            _client_id = int(_client_id)
            query = f"""
                SELECT client_id FROM oclienttraining
                WHERE experiment_id = {_experiment_id}
                    AND server_round = {_server_round}
                    AND client_id = {_client_id};
            """
        
            client_in_last_round = query_monetdb(conn, query)
            if client_in_last_round is None:
                # cliente caiu
                fails[_client_id] += 1
                
                while True:
                    
                    # subir novo cliente com esse client_id
                    ...

                    # esperar sua volta
                    query = f"""
                        SELECT COUNT(*) FROM idatasetload
                        WHERE experiment_id = {_experiment_id}
                            AND client_id = {_client_id};
                    """ 

                    count_dataset_load = query_monetdb(conn, query)
                    print(f"Client {_client_id} missing")
                    if count_dataset_load == (fails[_client_id] + 1):
                        # cliente já voltou
                        break
                    time.sleep(1)
        time.sleep(10)

if __name__ == "__main__":          
    main()