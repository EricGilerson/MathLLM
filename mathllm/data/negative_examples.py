"""Negative example generation: text containing numbers but no arithmetic.

These examples train the injection projection to output near-zero when the
hidden state contains numbers but no computation is expected. Without them,
the model hallucinates arithmetic into non-math contexts.

Categories mirror real-world text corpora: news, code, legal, medical,
academic, sports, recipes, logs, etc.
"""

from __future__ import annotations

import random

# Synthetic negative templates — numbers present but no computation
SYNTHETIC_NEGATIVE_TEMPLATES = [
    # ---- Aviation / Travel ----
    "Flight {a} departs from gate {b} at terminal {c}.",
    "Boarding pass: Seat {a}{letter}, Flight {b}, Gate {c}.",
    "The flight from JFK to LAX takes approximately {a} hours with a layover of {b} minutes.",
    "Train {a} arrives at platform {b} at {c}:{d} PM.",
    "Bus route {a} stops at station {b} every {c} minutes.",
    "Cruise ship deck {a}, cabin {b}. Departure: port {c}.",
    "Runway {a}L cleared for takeoff. Altitude: {b} feet.",
    "Connecting flight at gate {b}{letter}. Layover: {a} minutes.",
    "Highway {a} exits at mile marker {b}. Speed limit: {c} mph.",
    "Ferry terminal {a}. Departures at {c}:{d} and {c2}:{d2}.",
    # ---- Demographics / Statistics ----
    "The population in {year} was {pop}.",
    "Census data shows {a} households in district {b}.",
    "Approximately {a} people attended the event on {month} {day}.",
    "The city has {a} registered voters across {b} precincts.",
    "Survey of {a} respondents from {b} countries.",
    "Life expectancy in the region is {a} years. Median age: {b}.",
    "Unemployment rate: {a}.{b}%. Labor force: {c},{d2} people.",
    "District {b} has {a} schools serving {c},{d2} students.",
    "The county reports {a} births and {b} deaths this quarter.",
    "Voter turnout was {a}.{b}% in the {year} election.",
    # ---- References / Citations / Academic ----
    "See section {a}.{b}.{c} on page {d}.",
    "Reference: Chapter {a}, page {b}, paragraph {c}.",
    "As noted in [Figure {a}], the trend since {year} has been...",
    "Footnote {a}: Data collected between {year1} and {year2}.",
    "Table {a}.{b} shows the raw data.",
    "In Smith et al. ({year}), the authors analyzed {a} samples across {b} trials.",
    "According to Theorem {a}.{b}, the bound holds for all n > {c}.",
    "The experiment in Section {a} used {b} subjects over {c} weeks.",
    "Equation ({a}.{b}) defines the loss function. See Appendix {letter}.",
    "DOI: 10.{a}/{b}.{year}.{c}",
    "[{a}] Johnson, R. et al. \"Title.\" Journal, vol. {b}, no. {c}, {year}.",
    "As described in Definition {a}.{b}.{c}, the set S has cardinality at most n.",
    "Proposition {a}: For all primes p > {b}, the bound holds.",
    # ---- Technology / Software ----
    "Model GPT-{a} was trained on {b} billion tokens.",
    "Version {a}.{b}.{c} released on {month} {day}, {year}.",
    "The server runs on port {port} with {a} worker threads.",
    "GPU memory: {a} GB. VRAM utilization: {b}%.",
    "Batch size: {a}. Learning rate: 0.{b}. Epochs: {c}.",
    "Error code {a}: Connection timeout after {b} seconds.",
    "API rate limit: {a} requests per {b} seconds.",
    "Commit SHA: a{b}c{d}e. Branch: feature/{a}-update.",
    "Docker container {a} running on node {b}. Uptime: {c} hours.",
    "Latency: p50 = {a}ms, p99 = {b}ms. QPS: {c}.",
    "Model checkpoint at epoch {a}, step {b}. Loss: {c}.{d}.",
    "Cluster has {a} nodes with {b} CPUs and {c} GB RAM each.",
    "Build #{a} passed. {b} tests run, {c} skipped.",
    "Pull request #{a} merged by user {b}. {c} files changed.",
    "Issue #{a} opened {year}-{a:02d}-{b:02d}. Priority: P{d}.",
    "Deployment v{a}.{b}.{c} rolled out to {d}% of traffic.",
    # ---- Code with numeric constants ----
    "MAX_RETRIES = {a}",
    "port = {port}",
    "timeout_ms = {a}",
    "const BUFFER_SIZE = {a};",
    "private static final int THRESHOLD = {a};",
    "#define MAX_CONNECTIONS {a}",
    "config.max_workers = {a}",
    "DEFAULT_TIMEOUT = {a}",
    "BATCH_SIZE = {a}",
    "NUM_EPOCHS = {a}",
    "LEARNING_RATE = 0.{a}",
    "const PAGE_SIZE: usize = {a};",
    "static constexpr int kMaxRetries = {a};",
    "export const API_VERSION = {a};",
    "self.hidden_dim = {a}",
    "int fd = open(\"/dev/tty{a}\", O_RDWR);",
    "socket.bind(('0.0.0.0', {port}))",
    "logging.setLevel({a})",
    "cache_ttl_seconds = {a}",
    "retry_backoff_ms = {a}",
    # ---- Addresses / Identifiers ----
    "{a} Main Street, Apt {b}, Floor {c}.",
    "Order #{a}-{b}-{c} shipped on {month} {day}.",
    "Invoice #{a} for ${b}.{c}",
    "Patient ID: {a}{b}{c}. Room {d}.",
    "Case number: {year}-{a}-{b}.",
    "Serial number: SN-{a}-{b}-{c}.",
    "License plate: {letter}{a}{b}{c}.",
    "Tracking number: 1Z{a}{b}{c}{d}.",
    "ISBN: 978-{a}-{b}-{c}-{d}",
    "Social security ending in {a}. Account #{b}{c}.",
    "Parcel #{a} delivered to locker {b} at {c}:{d} PM.",
    "Badge ID: {a}{b}. Department: {c}.",
    "Ticket #TKT-{year}-{a}{b}. Row {c}, Seat {d}.",
    # ---- Dates and Times ----
    "Meeting scheduled for {month} {day}, {year} at {a}:{b} AM.",
    "The deadline is {month} {day}, {year}.",
    "Last updated: {year}-{a:02d}-{b:02d}.",
    "Event runs from {a}:{b} PM to {c}:{d} PM.",
    "Born on {month} {day}, {year}. Age: {a}.",
    "The conference is on {month} {day}-{day2}, {year}.",
    "Due date: {year}-{a:02d}-{b:02d}T{c}:{d}:00Z.",
    "Fiscal year {year}: Q1 through Q4.",
    "Sunrise at {a}:{b} AM. Sunset at {c}:{d} PM.",
    "Anniversary: {month} {day}, since {year1}.",
    # ---- Sports / Scores (mentioned but not computed) ----
    "Jersey number {a}, playing in position {b}.",
    "Season {year}: {a} wins, {b} losses.",
    "Stadium capacity: {a} seats. Attendance: {b}.",
    "Player #{a} has been with the team since {year}.",
    "Quarter {c}: Home {a}, Visitors {b}.",
    "Lap {a} of {b}. Current position: {c}th.",
    "Career stats: {a} games, {b} goals, {c} assists.",
    "Draft pick #{a} in round {b} of the {year} draft.",
    "Match point at {a}-{b}. Set {c} of {d}.",
    "Personal best: {a}.{b} seconds in the {c}m dash.",
    # ---- Measurements (not computed) ----
    "The building is {a} meters tall with {b} floors.",
    "Temperature: {a}°F. Humidity: {b}%.",
    "Weight: {a} kg. Height: {b} cm.",
    "Distance: {a} miles. Elevation: {b} feet.",
    "Dimensions: {a} x {b} x {c} inches.",
    "Wind speed: {a} knots. Visibility: {b} miles.",
    "Rainfall: {a}.{b} inches. Barometric pressure: {c}.{d} inHg.",
    "Latitude: {a}.{b}°N, Longitude: {c}.{d}°W.",
    "Magnitude {a}.{b} earthquake at depth {c} km.",
    "Fuel capacity: {a} gallons. Range: {b} miles.",
    "Wavelength: {a} nm. Frequency: {b} THz.",
    "Voltage: {a}V. Current: {b}mA. Resistance: {c} ohms.",
    # ---- Financial (mentioned, not computed) ----
    "Stock ticker: ${a}.{b}. Volume: {c}K.",
    "Account ending in {a}. Balance as of {month}: ${b}.{c}.",
    "Item SKU: {a}-{b}-{c}. Price: ${d}.99.",
    "Budget allocated: ${a},{b}.",
    "Market cap: ${a}.{b}B. P/E ratio: {c}.{d}.",
    "Revenue: ${a}M in Q{c} {year}.",
    "Bond yield: {a}.{b}%. Maturity: {year}.",
    "Lot size: {a} shares at ${b}.{c} per share.",
    "Invoice total: ${a},{b}.{c}. Due: {month} {day}.",
    "Tax ID: {a}-{b}. Fiscal code: {c}{d}.",
    # ---- Medical / Health ----
    "Patient age: {a}. Blood pressure: {b}/{c} mmHg.",
    "Dosage: {a}mg twice daily for {b} days.",
    "Heart rate: {a} bpm. SpO2: {b}%.",
    "Lab result: WBC {a}.{b}, RBC {c}.{d}.",
    "Prescription #{a}. Refills: {b}. Pharmacy code: {c}.",
    "BMI: {a}.{b}. Weight: {c} lbs.",
    "Room {a}, bed {b}. Admit date: {month} {day}, {year}.",
    "Temperature: {a}.{b}°C. Pulse: {c} bpm.",
    "Hemoglobin: {a}.{b} g/dL. Hematocrit: {c}%.",
    "MRI scan #{a}. Slice thickness: {b}mm.",
    # ---- Legal / Government ----
    "Case No. {year}-CV-{a}{b}. Filed in District {c}.",
    "Statute §{a}.{b}({c}). Effective date: {month} {day}, {year}.",
    "Exhibit {letter}-{a}. Bates number: {b}{c}{d}.",
    "Docket #{a}. Hearing scheduled for {month} {day}, {year}.",
    "Contract clause {a}.{b}: Termination after {c} days notice.",
    "Patent No. US{a},{b},{c}. Filed {year}.",
    "Regulation {a} CFR §{b}.{c}.",
    "Zoning code: R-{a}. Parcel ID: {b}-{c}-{d}.",
    "Ordinance #{a}-{year}. Effective {month} {day}, {year}.",
    "Permit #{a}{b}. Approved by inspector #{c}.",
    # ---- Recipes / Food ----
    "Preheat oven to {a}°F. Bake for {b} minutes.",
    "Add {a} cups of flour, {b} teaspoons of salt, and {c} eggs.",
    "Serves {a}. Prep time: {b} minutes. Cook time: {c} minutes.",
    "Mix {a} oz cream cheese with {b} tablespoons sugar.",
    "Simmer at {a}°F for {b} hours. Rest for {c} minutes.",
    "Calories: {a}. Protein: {b}g. Fat: {c}g. Carbs: {d}g.",
    "Use a {a}-inch pan. Grease with {b} tablespoons butter.",
    "Marinate for {a} hours. Grill at {b}°F for {c} minutes per side.",
    # ---- Logs / System Output ----
    "[{year}-{a:02d}-{b:02d} {c}:{d}:00] INFO: Request processed in {a}ms.",
    "PID {a}: Memory usage {b}MB / {c}MB.",
    "[Thread-{a}] Connected to {b}.{c}.{d}.{a}:{port}.",
    "ERROR {a}: File not found at line {b}.",
    "Retry {a}/{b}: Waiting {c} seconds before next attempt.",
    "Worker {a} processed {b} tasks in {c} seconds.",
    "[{a}] WARN: Queue depth at {b}. Threshold: {c}.",
    "Checkpoint saved at step {a}. ETA: {b} minutes.",
    # ---- Geography / Maps ----
    "Elevation: {a} ft above sea level. Population: {b}.",
    "ZIP code {a}{b}. Area code: ({c}).",
    "District {a}, ward {b}. Census tract: {c}.{d}.",
    "Coordinates: {a}°{b}'{c}\"N {a2}°{b}'{d}\"W.",
    "Route {a} connects town {b} to city {c}. Distance: {d} miles.",
    "Climate zone {a}. Average rainfall: {b} inches per year.",
    # ---- Education ----
    "Student ID: {a}{b}{c}. GPA: {a}.{b}. Credits: {c}.",
    "Course {a}{b}: Section {c}. Room {d}. Enrollment: {a}.",
    "Grade: {a}/{b}. Percentile: {c}th.",
    "Textbook ISBN: 978-{a}-{b}-{c}-{d}. Edition: {c}.",
    "Class of {year}. {a} graduates. Average SAT: {b}.",
    "Assignment #{a} due {month} {day}. Worth {b} points.",
    # ---- E-commerce / Retail ----
    "Product #{a}. {b} in stock. Rating: {c}.{d}/5.",
    "Order #{a}{b}: {c} items. Shipped via carrier #{d}.",
    "SKU-{a}-{b}. Aisle {c}, shelf {d}.",
    "Discount code: SAVE{a}. Valid until {month} {day}.",
    "Review #{a}: \"{b} stars. Would buy again.\"",
    "Cart total: {a} items. Estimated delivery: {b} days.",
    "Warehouse {a}. Bin location: {b}-{c}-{d}.",
    "Return authorization #{a}{b}. Reason code: {c}.",
    # ---- Music / Media ----
    "Track {a} of {b}. Duration: {c}:{d}.",
    "Album released {year}. {a} tracks. Runtime: {b} minutes.",
    "Episode {a}, Season {b}. Air date: {month} {day}, {year}.",
    "Page {a} of {b}. Chapter {c}: \"{d} Ways to...\"",
    "ISBN {a}-{b}-{c}. First printing: {year}. {d} pages.",
    "Podcast episode #{a}. Downloads: {b},{c}.",
    "Channel {a}. Subscribers: {b}K. Videos: {c}.",
]


class NegativeExampleSampler:
    """Generate text containing numbers but requiring no arithmetic."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
        self.letters = "ABCDEFGHJK"

    def _fill_template(self, template: str) -> str:
        """Fill a template with random plausible values."""
        replacements = {
            "{a}": str(self.rng.randint(1, 999)),
            "{b}": str(self.rng.randint(1, 999)),
            "{c}": str(self.rng.randint(1, 999)),
            "{d}": str(self.rng.randint(1, 99)),
            "{c2}": str(self.rng.randint(1, 12)),
            "{d2}": str(self.rng.randint(0, 999)),
            "{a2}": str(self.rng.randint(1, 180)),
            "{year}": str(self.rng.randint(1990, 2025)),
            "{year1}": str(self.rng.randint(1990, 2010)),
            "{year2}": str(self.rng.randint(2011, 2025)),
            "{pop}": f"{self.rng.randint(1, 999)},{self.rng.randint(0, 999):03d}",
            "{month}": self.rng.choice(self.months),
            "{day}": str(self.rng.randint(1, 28)),
            "{day2}": str(self.rng.randint(1, 28)),
            "{port}": str(self.rng.randint(1024, 65535)),
            "{letter}": self.rng.choice(self.letters),
            "{a:02d}": f"{self.rng.randint(1, 12):02d}",
            "{b:02d}": f"{self.rng.randint(1, 28):02d}",
        }
        result = template
        for key, value in replacements.items():
            result = result.replace(key, value)
        return result

    def sample(self, n: int) -> list[str]:
        """Generate n negative examples."""
        examples = []
        for _ in range(n):
            template = self.rng.choice(SYNTHETIC_NEGATIVE_TEMPLATES)
            examples.append(self._fill_template(template))
        return examples
