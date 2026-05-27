from schema import SessionLocal, Mudra

session = SessionLocal()

mudras = session.query(Mudra).all()

for m in mudras:
    print(m.id, m.name, m.description)

print("Total:", len(mudras))