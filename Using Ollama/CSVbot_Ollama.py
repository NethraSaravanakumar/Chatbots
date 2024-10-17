import langchain
from langchain.agents import create_spark_sql_agent
from langchain.agents.agent_toolkits import SparkSQLToolkit
from langchain.utilities.spark_sql import SparkSQL
from pyspark.sql import SparkSession
import ollama

# Create or get SparkSession
spark = SparkSession.builder.appName("LangChainExample").getOrCreate()

# Define schema and create database if not exists
schema = "langchain_examples"
spark.sql(f"CREATE DATABASE IF NOT EXISTS {schema}")
spark.sql(f"USE {schema}")

# Specify CSV file path and table name
csv_file_path = "sample.csv"
table = "sampletable2"

# Read CSV, infer schema, and save as a table
spark.read.csv(csv_file_path, header=True,
               inferSchema=True).write.saveAsTable(table)

# Show the content of the table
spark.table(table).show()

# Create Ollama chat model
llm = ollama.chat(model="mistral")

# Create SparkSQL instance
spark_sql = SparkSQL(schema=schema)

# Create SparkSQLToolkit instance
toolkit = SparkSQLToolkit(db=spark_sql, llm=llm)

# Create Spark SQL agent executor
agent_executor = create_spark_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

# Invoke a query
agent_executor.invoke("how many companies are there?")
