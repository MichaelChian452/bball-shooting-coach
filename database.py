import sqlite3

# Connect to database
conn = sqlite3.connect("shots.db")

# Create a cursor
c = conn.cursor()

# Create a Table
c.execute("""CREATE TABLE IF NOT EXISTS shots (
        release_time real, 
        release_angle real,
        elbow_angle real, 
        knee_angle real, 
        ball_position text
)""")
conn.commit()

# c.execute("INSERT INTO shots VALUES (555.0, 90.0, 90.0, 90.0, 'sheesh')")

# Each row is a tuple
for row in c.execute("SELECT rowid, * FROM shots"):
    print(row)
conn.commit()

conn.close()