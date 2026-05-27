from schema import SessionLocal, Mudra, init_db
import json
import os

MUDRA_INFO = {
"Alapadmam": {"meaning":"Fully bloomed lotus","description":"All fingers spread wide and curved outward like a blooming lotus.","difficulty":"Medium"},
"Anjali":{"meaning":"Salutation","description":"Both palms joined evenly showing respect or greeting.","difficulty":"Easy"},
"Aralam":{"meaning":"Drinking nectar","description":"Index finger bent while other fingers remain straight.","difficulty":"Medium"},
"Ardhachandran":{"meaning":"Half moon","description":"All fingers together with thumb stretched outward.","difficulty":"Easy"},
"Ardhapathaka":{"meaning":"Half flag","description":"Little finger bent while others remain straight.","difficulty":"Medium"},
"Berunda":{"meaning":"Two-headed bird","description":"Both hands joined with interlocked thumbs forming a mythical bird.","difficulty":"Hard"},
"Bramaram":{"meaning":"Bee","description":"Thumb and middle finger touch while index bends inward.","difficulty":"Medium"},
"Chakra":{"meaning":"Wheel","description":"Both palms facing each other forming circular shape.","difficulty":"Easy"},
"Chandrakala":{"meaning":"Crescent moon","description":"Index finger extended with thumb stretched sideways.","difficulty":"Easy"},
"Chaturam":{"meaning":"Small quantity","description":"Thumb touches base of ring finger with others straight.","difficulty":"Hard"},
"Garuda":{"meaning":"Eagle","description":"Thumbs interlocked with fingers spread like wings.","difficulty":"Medium"},
"Hamsapaksha":{"meaning":"Swan wing","description":"All fingers straight together with little finger extended.","difficulty":"Medium"},
"Hamsasyam":{"meaning":"Swan beak","description":"Thumb and index finger joined forming small circle.","difficulty":"Easy"},
"Kangulam":{"meaning":"Bell shape","description":"Ring finger bent inward while others curved.","difficulty":"Hard"},
"Kapith":{"meaning":"Holding flowers","description":"Index finger bent over thumb in a closed formation.","difficulty":"Medium"},
"Kapotham":{"meaning":"Pigeon","description":"Both hands in Anjali with slight hollow between palms.","difficulty":"Easy"},
"Karkatta":{"meaning":"Crab","description":"Fingers of both hands interlocked.","difficulty":"Easy"},
"Kartariswastika":{"meaning":"Crossed scissors","description":"Hands crossed using Kartari formation.","difficulty":"Medium"},
"Katakamukha_1":{"meaning":"Plucking flowers","description":"Thumb, index and middle finger joined.","difficulty":"Medium"},
"Katakamukha_2":{"meaning":"Holding garland","description":"Thumb, index and middle finger joined slightly curved.","difficulty":"Medium"},
"Katakamukha_3":{"meaning":"Drawing bow","description":"Thumb, index and middle joined with hand extended.","difficulty":"Medium"},
"Katakavardhana":{"meaning":"Marriage","description":"Both hands in Katakamukha crossed at wrists.","difficulty":"Medium"},
"Katrimukha":{"meaning":"Scissors","description":"Index and middle finger separated forming V shape.","difficulty":"Medium"},
"Khatva":{"meaning":"Bed or throne","description":"Both hands forming flat layered structure.","difficulty":"Hard"},
"Kilaka":{"meaning":"Love or affection","description":"Little fingers of both hands joined.","difficulty":"Easy"},
"Kurma":{"meaning":"Tortoise","description":"Hands layered forming shell shape.","difficulty":"Medium"},
"Matsya":{"meaning":"Fish","description":"Hands overlapped sideways with thumbs extended.","difficulty":"Easy"},
"Mayura":{"meaning":"Peacock","description":"Ring finger touches thumb with others extended.","difficulty":"Medium"},
"Mrigasirsha":{"meaning":"Deer head","description":"Thumb and little finger extended with middle fingers bent.","difficulty":"Hard"},
"Mukulam":{"meaning":"Bud","description":"All fingers joined at tips forming cone shape.","difficulty":"Easy"},
"Mushti":{"meaning":"Fist","description":"All fingers curled tightly into palm.","difficulty":"Easy"},
"Nagabandha":{"meaning":"Serpent tie","description":"Hands intertwined forming snake pattern.","difficulty":"Hard"},
"Padmakosha":{"meaning":"Lotus bud","description":"Fingers curved slightly forming cup shape.","difficulty":"Medium"},
"Pasha":{"meaning":"Noose","description":"Index fingers hooked together.","difficulty":"Medium"},
"Pathaka":{"meaning":"Flag","description":"All fingers straight and together.","difficulty":"Easy"},
"Pushpaputa":{"meaning":"Offering flowers","description":"Both palms cupped together.","difficulty":"Easy"},
"Sakata":{"meaning":"Demon","description":"Index and little fingers extended like horns.","difficulty":"Medium"},
"Samputa":{"meaning":"Container","description":"Hands cupped forming closed hollow.","difficulty":"Easy"},
"Sarpasirsha":{"meaning":"Snake head","description":"Fingers together curved forward.","difficulty":"Medium"},
"Shanka":{"meaning":"Conch","description":"One hand wraps around thumb of other.","difficulty":"Medium"},
"Shivalinga":{"meaning":"Shiva symbol","description":"One fist placed over upright palm.","difficulty":"Easy"},
"Shukatundam":{"meaning":"Parrot beak","description":"Index finger bent with thumb supporting.","difficulty":"Medium"},
"Sikharam":{"meaning":"Peak","description":"Fist with thumb extended upward.","difficulty":"Easy"},
"Simhamukham":{"meaning":"Lion face","description":"Middle and ring fingers touch thumb.","difficulty":"Hard"},
"Suchi":{"meaning":"Needle","description":"Index finger pointing upward.","difficulty":"Easy"},
"Swastikam":{"meaning":"Auspicious cross","description":"Hands crossed at wrists.","difficulty":"Easy"},
"Tamarachudam":{"meaning":"Rooster crest","description":"Index bent over thumb with fist closed.","difficulty":"Medium"},
"Tripathaka":{"meaning":"Three parts of flag","description":"Ring finger bent others straight.","difficulty":"Easy"},
"Trishulam":{"meaning":"Trident","description":"Three fingers extended with thumb holding little.","difficulty":"Medium"},
"Varaha":{"meaning":"Boar","description":"Hands joined forming snout shape.","difficulty":"Medium"}
}

def seed_mudras():
    session = SessionLocal()

    # remove previous data
    print("Deleting old data...")
    session.query(Mudra).delete()
    session.commit()

    # load class order
    with open("models/class_names.json", "r") as f:
        class_names = json.load(f)

    print("Seeding", len(class_names), "mudras")

    for name in class_names:
        info = MUDRA_INFO.get(name, {})

        mudra = Mudra(
            name=name,
            meaning=info.get("meaning", ""),
            description=info.get("description", ""),
            hand_formation=info.get("description", ""),
            usage="Used in Bharatanatyam expression.",
            common_mistakes="Incorrect finger alignment.",
            improvement_tips="Practice slowly with mirror.",
            difficulty=info.get("difficulty", "Medium")
        )

        session.add(mudra)

    session.commit()
    session.close()

    print("Database seeded successfully!")


if __name__ == "__main__":
    init_db()
    seed_mudras()