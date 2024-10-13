# How to Use Hadoop with This Project

### Start the Hadoop Server

1. Format if necessary:
    ```bash
    sbin/hadoop namenode -format
    ```

2. Start the Hadoop Distributed File System (HDFS):
    ```bash
    sbin/start-dfs.cmd
    ```

3. Start YARN:
    ```bash
    sbin/start-yarn.cmd
    ```

4. Check running Java processes:
    ```bash
    jps
    ```

### How to Upload Files from the Command Line

1. Upload `movies.csv` to HDFS:
    ```bash
    hadoop fs -put C:/path/to/your/movies.csv /path/in/hdfs/movies.csv
    ```

2. Upload `ratings.csv` to HDFS:
    ```bash
    hadoop fs -put C:/path/to/your/ratings.csv /path/in/hdfs/ratings.csv
    ```

*(Replace `path/to/your` with the actual file path)*

### How to Access the Files Uploaded in HDFS

1. Install the `hdfs` package:
    ```bash
    pip install hdfs
    ```

2. Example Python code to read the file from HDFS:

    ```python
    from hdfs import InsecureClient
    import pandas as pd

    # HDFS configuration (Adjust the URL to match your HDFS setup)
    HDFS_URL = 'http://localhost:9870'  # HDFS Web UI address
    HDFS_CLIENT = InsecureClient(HDFS_URL, user='hadoop')  # 'hadoop' is the username

    # HDFS file path
    hdfs_file_path = '/movies.csv'

    # Function to read CSV from HDFS
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
    ```
