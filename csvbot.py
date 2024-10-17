import langchain
from langchain.agents import create_spark_sql_agent
from langchain.agents.agent_toolkits import SparkSQLToolkit
from langchain.utilities.spark_sql import SparkSQL
from pyspark.sql import SparkSession
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import BaseModel

spark = SparkSession.builder.getOrCreate()
schema = "langchain_example"
spark.sql(f"CREATE DATABASE IF NOT EXISTS {schema}")
spark.sql(f"USE {schema}")
csv_file_path = "Prediction_Data.csv"
table = "Prediction7"
spark.read.csv(csv_file_path, header=True, inferSchema=True).write.saveAsTable(table)
spark.table(table).show()

llm = LlamaCpp(
                model_path="zephyr-7b-alpha.Q8_0.gguf",
                n_ctx=4000,
                temperature=0.0,
                max_tokens=4000,
                top_p=1,
                n_batch=512,
                n_gpu_layers=400,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                verbose=True,
            )

spark_sql = SparkSQL(schema=schema)
toolkit = SparkSQLToolkit(db=spark_sql, llm=llm)
agent_executor = create_spark_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

agent_executor.run("How many company are there in prediction7 table?")
agent_executor.run("In the Prediction7 table what is the predicted value of quantity sold of Bestrun1 in 2024 ?")


