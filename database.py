import sqlite3

# Connect to database
conn = sqlite3.connect('shots.db')

# Create a cursor
c = conn.cursor()

# Create a Table
c.execute("""CREATE TABLE customers (
          shot_number DATATYPE, 
          release_time DATATYPE, 
          release_angle DATATYPE,
          elbow_angles DATATYPE, 
          shoulder_angles DATATYPE, 
          knee_angles DATATYPE
)""")