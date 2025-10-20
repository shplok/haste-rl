from haste_env import HasteEnv
import time

env = HasteEnv()

print("testing health detection")
print("take damage in the game and watch output...")

for i in range(100):
    health = env._read_health_quarters()
    print(f"step {i}: health quarters = {health}/4")
    time.sleep(0.5)

env.close()