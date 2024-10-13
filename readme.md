***HOW TO USE HADOOP WITH THIS PROJECT***

**Start the hadoop server**
Format if necessary
# sbin> hadoop namenode -format
# sbin> start-dfs.cmd
# sbin> start-yarn.cmd
# sbin> jps

**How upload the files from the command line**
# sbin> hadoop fs -put C:/path/to/your/movies.csv /path/in/hdfs/movies.csv
# sbin> hadoop fs -put C:/path/to/your/ratings.csv /path/in/hdfs/ratings.csv
*replace path/to/your with actual path*

**How access the files uploaded in the hdfs**
# pip install hdfs
----------------example snippet---------------------------------------- 

from hdfs import InsecureClient
import pandas as pd

# HDFS configuration (Adjust the URL to match your HDFS setup)
HDFS_URL = 'http://localhost:9870'  # HDFS Web UI address
HDFS_CLIENT = InsecureClient(HDFS_URL, user='hadoop')  # 'hadoop' is the username

# HDFS file path
***(root dir in this example)***
hdfs_file_path = '/movies.csv'      

# Function to read CSV from HDFS #
def read_csv_from_hdfs(hdfs_file_path):
    try:
        # Read the CSV file from HDFS
        with HDFS_CLIENT.read(hdfs_file_path, encoding='utf-8') as reader:
            # Load the CSV data into a Pandas DataFrame
            df = pd.read_csv(reader)
        return df
    except Exception as e:
        print(f"Error reading file from HDFS: {e}")
        return None
----------------------snippert end---------------------------------------