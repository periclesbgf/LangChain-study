from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Date, Numeric, ForeignKey
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

user = os.getenv("POSTGRES_USER")
password = os.getenv("PASSWORD")
host = os.getenv("HOST")
port = os.getenv("PORT")
database = os.getenv("DATABASE")

engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')
metadata = MetaData()

chassis_table = Table('Chassis', metadata,
    Column('id', Integer, primary_key=True),
    Column('Chassi', Integer, unique=True),
    Column('Modelo', Integer),
    Column('Cliente', Integer),
    Column('Contrato', Integer)
)

telemetria_table = Table('Telemetria', metadata,
    Column('id', Integer, primary_key=True),
    Column('Chassi', Integer, ForeignKey('Chassis.Chassi')),
    Column('UnidadeMedida', String),
    Column('Categoria', String),
    Column('Data', Date),
    Column('Serie', String),
    Column('Valor', Numeric)
)

metadata.drop_all(engine)
metadata.create_all(engine)

file_path = '/Users/peric/projects/LangChain-study/app/database/data/Bases_case_final.xlsx'
df_chassis = pd.read_excel(file_path, sheet_name='Chassis')
df_telemetria = pd.read_excel(file_path, sheet_name='Telemetria')

print("Columns in df_chassis:", df_chassis.columns)
print("Columns in df_telemetria:", df_telemetria.columns)

df_chassis.to_sql('Chassis', con=engine, if_exists='append', index=False)
df_telemetria.to_sql('Telemetria', con=engine, if_exists='replace', index=False)

print("Data uploaded successfully!")
