"""
Generate a synthetic CRM dataset (mixed text / numeric / categorical / datetime).
Usage: python scripts/make_crm.py [n_rows] [out_path]
Attributes are correlated with the free-text notes so embeddings have real signal to learn.
"""
import csv, random, sys
from datetime import datetime, timedelta

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/crm.csv"
random.seed(42)
TODAY = datetime(2026, 10, 1)

FIRST = "Aarav Priya Rohan Sita Hari Maya Liam Olivia Noah Emma Mateo Sofia Chen Mei Yuki Hiro Fatima Omar Lars Ingrid Pierre Chloe Amara Kwame Diego Lucia Ivan Anya James Sarah Ahmed Leila Ravi Anita Tomas Eva".split()
LAST = "Sharma Rai Thapa Gurung Smith Johnson Garcia Martinez Wang Li Tanaka Sato Khan Hussain Nielsen Larsen Dubois Martin Okafor Mensah Silva Rossi Petrov Ivanova Brown Davis Patel Singh Kim Nguyen Muller Schmidt".split()

LOCS = [("Kathmandu","Bagmati","Nepal","Asia/Kathmandu"),("Pokhara","Gandaki","Nepal","Asia/Kathmandu"),
 ("Mumbai","Maharashtra","India","Asia/Kolkata"),("Bengaluru","Karnataka","India","Asia/Kolkata"),
 ("London","England","UK","Europe/London"),("Berlin","Berlin","Germany","Europe/Berlin"),
 ("Paris","Ile-de-France","France","Europe/Paris"),("New York","NY","USA","America/New_York"),
 ("Austin","TX","USA","America/Chicago"),("San Francisco","CA","USA","America/Los_Angeles"),
 ("Toronto","ON","Canada","America/Toronto"),("Sydney","NSW","Australia","Australia/Sydney"),
 ("Singapore","Singapore","Singapore","Asia/Singapore"),("Tokyo","Tokyo","Japan","Asia/Tokyo"),
 ("Lagos","Lagos","Nigeria","Africa/Lagos"),("Sao Paulo","SP","Brazil","America/Sao_Paulo")]

INDUSTRIES = {
 "Healthcare":["clinic scheduling","patient records","HIPAA compliance","telehealth"],
 "Retail":["inventory sync","point-of-sale","loyalty program","seasonal promotions"],
 "Finance":["fraud detection","audit reporting","KYC onboarding","regulatory compliance"],
 "Education":["student portal","course analytics","online exams","enrollment automation"],
 "Logistics":["fleet tracking","route optimization","warehouse automation","delivery SLAs"],
 "SaaS":["API integration","usage-based billing","customer onboarding","churn analytics"],
 "Manufacturing":["supply chain visibility","quality control","predictive maintenance","ERP integration"],
 "Hospitality":["booking engine","guest feedback","revenue management","staff rostering"]}

SUFFIX = ["Labs","Group","Solutions","Partners","Systems","Holdings","Works","Ltd","Co","Technologies"]
TITLES = ["CEO","CTO","COO","VP of Sales","Head of Operations","IT Manager","Procurement Lead","Data Analyst","Product Manager","Founder","Office Manager","Marketing Director"]
TIERS = {"Free":(0,0),"Starter":(30,120),"Professional":(150,600),"Enterprise":(800,5000)}
CHANNELS = ["email","phone","whatsapp","sms","in-person"]
SOURCES = ["organic search","referral","paid ads","conference","partner","cold outreach","webinar"]

TICKET = {
 "billing":["Invoice shows wrong amount","Double charged this month","Request for refund","Need updated tax invoice"],
 "technical":["API returns 500 errors","Dashboard loads very slowly","Integration with ERP failing","Cannot reset two-factor auth"],
 "feature":["Request: bulk CSV export","Asking for SSO support","Wants custom reporting module","Mobile app dark mode"],
 "onboarding":["Need help with initial setup","Training session request","Data migration questions","How to add team members"],
 "none":[""]}

HAPPY = ["Very satisfied with the service and praised the support team.","Enthusiastic about the product; open to a case study.","Reports big efficiency gains and wants to expand usage.","Regular, friendly contact; renewal looks safe."]
NEUTRAL = ["Uses the product steadily but has not explored advanced features.","Neutral feedback; asked for a pricing comparison.","Occasional questions, no major concerns raised.","Satisfied overall but price-conscious."]
UNHAPPY = ["Frustrated by repeated outages and slow responses.","Threatened to cancel unless the billing issue is resolved.","Evaluating a competitor after poor onboarding experience.","Complained about missing features promised during the sales process."]

def rdate(a, b): return a + timedelta(seconds=random.randint(0, int((b-a).total_seconds())))
def iso(d): return d.strftime("%Y-%m-%d %H:%M:%S") if d else ""

rows = []
for i in range(N):
    fn, ln = random.choice(FIRST), random.choice(LAST)
    city, state, country, tz = random.choice(LOCS)
    industry = random.choice(list(INDUSTRIES))
    topics = random.sample(INDUSTRIES[industry], 2)
    company = f"{random.choice(LAST)} {random.choice(SUFFIX)}"
    tier = random.choices(list(TIERS), [30,30,25,15])[0]
    signup = rdate(datetime(2018,1,1), TODAY - timedelta(days=14))
    tenure_days = (TODAY - signup).days
    # latent satisfaction drives numbers + text
    sat = max(0, min(10, round(random.gauss(7, 2.2))))
    mood = HAPPY if sat >= 8 else NEUTRAL if sat >= 5 else UNHAPPY
    lo, hi = TIERS[tier]
    mrr = round(random.uniform(lo, hi), 2) if hi else 0.0
    orders = max(0, int(random.gauss(tenure_days/60 + sat, 4))) if tier != "Free" else random.randint(0, 3)
    ltv = round(mrr * tenure_days/30 * random.uniform(0.6, 1.0), 2)
    aov = round(ltv/orders, 2) if orders else 0.0
    churn = min(0.99, max(0.01, round((10-sat)/12 + random.gauss(0,0.08) + (0.1 if tier=="Free" else 0), 3)))
    last_purchase = rdate(max(signup, TODAY - timedelta(days=int(30 + churn*600))), TODAY) if orders else None
    last_contact = rdate(max(signup, TODAY - timedelta(days=int(10 + churn*300))), TODAY)
    cat = random.choices(list(TICKET), [3,3,2,2,5])[0] if sat < 8 else random.choices(list(TICKET), [1,1,2,1,9])[0]
    subject = random.choice(TICKET[cat])
    open_tix = random.randint(1,4) if cat != "none" and sat < 5 else (1 if cat != "none" and random.random() < .3 else 0)
    stage = ("churned" if churn > .8 and random.random() < .6 else "at_risk" if churn > .55 else "active")
    if tier == "Free" and orders == 0: stage = "lead"
    title, source, channel = random.choice(TITLES), random.choice(SOURCES), random.choice(CHANNELS)
    article = "an" if industry[0] in "AEIOU" else "a"
    notes = (f"{fn} is {random.choice(['the','a'])} {title} at {article} {industry.lower()} company in {city}. "
             f"Main interests: {topics[0]} and {topics[1]}. {random.choice(mood)} "
             + (f"Latest ticket: {subject.lower()}. " if subject else "No open issues. ")
             + f"Found us via {source}; prefers {channel}.")
    dob = rdate(datetime(1960,1,1), datetime(2002,12,31)).date()
    phone = f"+{random.choice([977,91,44,49,33,1,61,65,81,234,55])}-{random.randint(9000000000,9999999999)}"
    rows.append(dict(customer_id=f"CUST-{i+1:05d}", first_name=fn, last_name=ln, email=f"{fn}.{ln}{random.randint(1,99)}@{company.split()[0].lower()}.com".lower(),
        phone=phone, date_of_birth=dob, city=city, state=state, country=country, timezone=tz, company=company, industry=industry,
        job_title=title, plan_tier=tier, lifecycle_stage=stage, acquisition_source=source,
        preferred_channel=channel, signup_date=iso(signup), last_purchase_date=iso(last_purchase),
        last_contact_date=iso(last_contact), total_orders=orders, monthly_recurring_revenue=mrr, lifetime_value=ltv,
        avg_order_value=aov, satisfaction_score=sat, churn_risk=churn, open_tickets=open_tix, last_ticket_category=cat if cat!="none" else "",
        last_ticket_subject=subject, interests="; ".join(topics), newsletter_opt_in=random.random()<.6, notes=notes))

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(f"wrote {len(rows)} rows -> {OUT}")
