from pymongo import MongoClient
import json
import subprocess

client = MongoClient('localhost', 27017)
db = client['flowerprov']

status = db.command("serverStatus")

with open('server_status.json', 'w') as outfile:
    json.dump(status, outfile, default=str)


checkpoints_stats = db.command("collstats", "checkpoints")
with open('checkpoints_stats.json', 'w') as outfile:
    json.dump(checkpoints_stats, outfile, default=str)



cmd = """ps -eo size,pid,user,command --sort -size | \
    awk '{ hr=$1/1024 ; printf("%13.2f Mb ",hr) } { for ( x=4 ; x<=NF ; x++ ) { printf("%s ",$x) } print "" }' |\
    cut -d "" -f2 | cut -d "-" -f1"""

with open('memory_usage.txt', 'w') as outfile:
    outfile.write( subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read().decode('utf-8'))
