from langchain_community.utilities.sql_database import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///../../../../Chinook.db")

print(db.get_table_info())

# if __name__ == "__main__":
#     main()
