from datetime import datetime
from moodforge_main import generate_daily  # fake6.py dosyanla aynı dizinde olmalı

BASE_DIR = "test_output"
num_trials = 5  

print("\n🔬 GRADE I ve GRADE III ÜRETİM TESTİ BAŞLIYOR...\n")

# 🟢 Grade I (Normal birey)
for i in range(num_trials):
    pid = f"test_G1_{i:02d}"
    print(f"🟢 Test G1-{i+1}: Grade 'I' zorlanıyor...")
    generate_daily(
        pid=pid,
        date=datetime.today(),
        disordered=False,
        disorder_type=None,
        forced_grade="I"
    )

# 🟡 Grade III (Disordered birey)
for i in range(num_trials):
    pid = f"test_G3_{i:02d}"
    print(f"🟡 Test G3-{i+1}: Grade 'III' zorlanıyor...")
    generate_daily(
        pid=pid,
        date=datetime.today(),
        disordered=True,
        disorder_type="Depresyon",  # örnek bozukluk
        forced_grade="III"
    )
