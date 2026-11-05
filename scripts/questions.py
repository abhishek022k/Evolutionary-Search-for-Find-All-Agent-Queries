"""The candidate "find all" questions. Each rule set is the agreed definition."""

QUESTIONS = {
    "stay_home_boom": {
        "question": (
            "Does the filing state that individual consumers' pandemic lifestyle "
            "changes, such as spending more time at home or taking up home, hobby or "
            "outdoor activities, led them to buy or use more of the company's products "
            "or services?"
        ),
        "yes": (
            "The company reports this actually happened, and says individuals or "
            "households bought or used more because of how their daily life changed. "
            "Buying more and using more both count. Remote work or distance learning "
            "counts only when individuals or households are the ones using more. "
            "Example: \"Sales of bicycles surged as families sought outdoor activities "
            "during lockdowns.\""
        ),
        "no": (
            "Hypothetical language (\"could increase\"). Stay-at-home orders that closed "
            "stores or hurt the business. Businesses, governments or other organizations "
            "buying or using more, even if their staff worked remotely. Higher demand "
            "with no stated link to how consumers changed their behavior."
        ),
    },
    "new_offering": {
        "question": (
            "Does the filing state that the company launched or began offering a new "
            "product or service specifically in response to COVID-19?"
        ),
        "yes": (
            "The offering is new and the filing ties it to the pandemic. New ways of "
            "delivering a service count, such as a restaurant adding curbside pickup "
            "or a clinic adding telehealth. Example: \"We began producing hand "
            "sanitizer in response to the pandemic.\""
        ),
        "no": (
            "Expanding something the company already offered. Plans or possibilities "
            "(\"we may launch\"). A new offering not linked to the pandemic."
        ),
    },
    "panic_buying": {
        "question": (
            "Does the filing state that consumers stockpiled, panic-bought or bought "
            "unusually large quantities because of COVID-19, affecting the company's "
            "sales or inventory?"
        ),
        "yes": (
            "The company reports this consumer behavior actually happened. Example: "
            "\"Consumers stocked up on pantry staples in March, lifting our sales.\""
        ),
        "no": (
            "Hypothetical language. The company building its own safety stock. Higher "
            "demand with no mention of consumers stocking up."
        ),
    },
}
