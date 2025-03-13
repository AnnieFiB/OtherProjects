import os
from sqlalchemy import create_engine, Column, Integer, String, Float, func, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# Define the SQLAlchemy engine and Base
Base = declarative_base()

# Define the Student Table
class StudentMentalHealthData(Base):
    __tablename__ = 'students_mental_health_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_type = Column(String)  # International or Domestic student
    japanese_proficiency = Column(String)  # Japanese language proficiency
    english_proficiency = Column(String)  # English language proficiency
    academic_level = Column(String)  # Academic level (undergraduate/graduate)
    age = Column(Integer)  # Student age
    length_of_stay = Column(Float)  # Length of stay in years
    PHQ9_depression_score = Column(Integer)  # PHQ-9 depression score
    social_connectedness_score = Column(Integer)  # Social connectedness score (SCS test)
    acculturative_stress_score = Column(Integer)  # Acculturative stress score (ASISS test)

# Define the engine and Session globally
engine = None
Session = None

def setup_database(db_name):
    """
    Set up the database by checking if it exists, dropping it if it does, and creating a new one.

    Parameters:
    db_name (str): The name of the SQLite database file (e.g., 'students_mental_health.db').
    """
    global engine, Session  # Use global variables for engine and Session

    # Construct the database path
    DATABASE_PATH = f'sqlite:///{db_name}'
    db_file = db_name  # SQLite database file path

    # Check if the database file exists
    if os.path.exists(db_file):
        print(f"⚠ Database '{db_file}' already exists. Dropping it...")
        drop_database(db_name)  # Drop the existing database

    # Create a new database
    engine = create_engine(DATABASE_PATH, echo=False)
    Base.metadata.create_all(engine)  # Create the table in the database
    Session = sessionmaker(bind=engine)  # Create a session factory
    print(f"✅ Database '{db_name}' created successfully.")


def setup_and_insert_data(df, db_name):
    """
    Set up the database, create the table, and insert data from the provided DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing student data.
    db_name (str): The name of the SQLite database file (e.g., 'students_mental_health.db').
    """
    setup_database(db_name)  # Set up the database
    session = Session()
    try:
        print("Debug: Starting to insert data.")  # Debug print
        for index, row in df.iterrows():
            student = StudentMentalHealthData(
                student_type=row['inter_dom'],
                japanese_proficiency=row['japanese_cate'],
                english_proficiency=row['english_cate'],
                academic_level=row['academic'],
                age=row['age'],
                length_of_stay=row['stay'],
                PHQ9_depression_score=row['todep'],
                social_connectedness_score=row['tosc'],
                acculturative_stress_score=row['toas']
            )
            session.add(student)

        print("Debug: About to commit the transaction.")  # Debug print
        session.commit()
        print("✅ Data insertion completed successfully.")
    except Exception as e:
        print(f"⚠ Error: {e}")
    finally:
        session.close()
        print("Debug: Session closed.")  # Debug print



def execute_query(query):
    """
    Execute a raw SQL query and return the results.

    Parameters:
    query (str): The SQL query to execute.

    Returns:
    list: A list of tuples containing the query results.
    """
    global engine  # Ensure engine is accessible
    with engine.connect() as connection:
        result = connection.execute(text(query))  # Execute the raw SQL query
        return result.fetchall()  # Return the query resultsss

def get_unique_values(column_name):
    """
    Get unique values from a specified column in the students_mental_health_data table.

    Parameters:
    column_name (str): The name of the column to get unique values from.

    Returns:
    list: A list of unique values from the specified column.
    """
    global engine  # Ensure engine is accessible
    query = text(f"SELECT DISTINCT {column_name} FROM students_mental_health_data")
    with engine.connect() as connection:
        result = connection.execute(query)
        unique_values = [row[0] for row in result]  # Extract unique values from the result
    return unique_values

def get_summary_statistics():
    """
    Calculate and return summary statistics for PHQ-9, SCS, and ASISS scores.

    Returns:
    dict: A dictionary containing min, max, and avg values for each score.
    """
    global Session  # Ensure Session is accessible
    session = Session()
    try:
        # Define the query for summary statistics
        summary_query = session.query(
            func.round(func.min(StudentMentalHealthData.PHQ9_depression_score), 2).label('min_phq'),
            func.round(func.max(StudentMentalHealthData.PHQ9_depression_score), 2).label('max_phq'),
            func.round(func.avg(StudentMentalHealthData.PHQ9_depression_score), 2).label('avg_phq'),
            func.round(func.min(StudentMentalHealthData.social_connectedness_score), 2).label('min_scs'),
            func.round(func.max(StudentMentalHealthData.social_connectedness_score), 2).label('max_scs'),
            func.round(func.avg(StudentMentalHealthData.social_connectedness_score), 2).label('avg_scs'),
            func.round(func.min(StudentMentalHealthData.acculturative_stress_score), 2).label('min_asiss'),
            func.round(func.max(StudentMentalHealthData.acculturative_stress_score), 2).label('max_asiss'),
            func.round(func.avg(StudentMentalHealthData.acculturative_stress_score), 2).label('avg_asiss')
        )

        # Execute the query
        summary_stats = summary_query.one()

        # Return the results as a dictionary
        return {
            'min_phq': summary_stats.min_phq,
            'max_phq': summary_stats.max_phq,
            'avg_phq': summary_stats.avg_phq,
            'min_scs': summary_stats.min_scs,
            'max_scs': summary_stats.max_scs,
            'avg_scs': summary_stats.avg_scs,
            'min_asiss': summary_stats.min_asiss,
            'max_asiss': summary_stats.max_asiss,
            'avg_asiss': summary_stats.avg_asiss
        }
    except Exception as e:
        print(f"⚠ Error: {e}")
        return None
    finally:
        # Close the session
        session.close()

def drop_database(db_name):
    """
    Drop the SQLite database by closing all connections and deleting the database file.

    Parameters:
    db_name (str): The name of the SQLite database file (e.g., 'students_mental_health.db').
    """
    global engine, Session  # Use global variables for engine and Session

    # Close all sessions and dispose of the engine
    Session.close_all()  # Close all sessions
    engine.dispose()  # Dispose of the engine to close all connections
    print("✅ All database connections and sessions closed.")

    # Check if the file exists
    if os.path.exists(db_name):
        try:
            os.remove(db_name)
            print(f"✅ Database '{db_name}' dropped successfully.")
        except PermissionError as e:
            print(f"⚠ Error: {e}. The database file is still in use.")
    else:
        print(f"⚠ Database '{db_name}' does not exist.")