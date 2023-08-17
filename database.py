import sqlite3

# Connect to database
conn = sqlite3.connect("shots.db")

# Create a cursor
c = conn.cursor()

# Create a Table
# c.execute("""CREATE TABLE IF NOT EXISTS shots (
#         release_time real, 
#         release_angle real,
#         elbow_angle real, 
#         knee_angle real, 
#         ball_position text
# )""")
# conn.commit()

# c.execute("INSERT INTO shots VALUES (555.0, 90.0, 90.0, 90.0, 'sheesh')")

# Each row is a tuple
c.execute("SELECT * FROM shots ORDER BY rowid DESC LIMIT 1")
result = c.fetchone()
print(result)

POSITION_OF_BALLPOSITIONS_IN_DB = 4

print(result[POSITION_OF_BALLPOSITIONS_IN_DB])
result_list = result[POSITION_OF_BALLPOSITIONS_IN_DB][1:-2].split('], ')
to_return = []
for position in result_list:
    ball_position = list(map(lambda num: float(num), position[1:].split(', ')))
    to_return.append(ball_position)

print(to_return)

conn.close()