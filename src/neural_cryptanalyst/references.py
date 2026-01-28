REFERENCES = {
    "picek2023": {
        "authors": "Picek, S., Perin, G., Mariot, L., Wu, L., & Batina, L.",
        "title": "SoK: Deep Learning-based Physical Side-channel Analysis",
        "venue": "ACM Computing Surveys",
        "year": 2023,
        "doi": "10.1145/3569577",
    },
    "bursztein2024": {
        "authors": "Bursztein, E., Invernizzi, L., Král, K., Moghimi, D., Picod, J.-M., & Zhang, M.",
        "title": "Generalized Power Attacks against Crypto Hardware using Long-Range Deep Learning",
        "venue": "IACR Transactions on Cryptographic Hardware and Embedded Systems",
        "year": 2024,
        "doi": "10.46586/tches.v2024.i3.472-499",
    },
    "masure2020": {
        "authors": "Masure, L., Dumas, C., & Prouff, E.",
        "title": "A Comprehensive Study of Deep Learning for Side-Channel Analysis",
        "venue": "IACR TCHES",
        "year": 2020,
        "doi": "10.13154/tches.v2020.i1.348-375",
    },
}

def cite(key: str) -> str:
    ref = REFERENCES.get(key)
    if ref:
        return f"{ref['authors']} ({ref['year']})"
    return "[CITATION NEEDED]"
