disease_info = {
    "Early_Leaf_Spot": {
        "cause": "Caused by the fungus *Cercospora arachidicola*, usually appearing in warm and wet weather.",
        "solution": """To effectively control early leaf spot, action must be taken as soon as the first signs appear. 
Dark brown spots with yellow halos on leaves are the key indicator. 

Start spraying fungicides like **chlorothalonil** or **mancozeb**, ensuring thorough leaf coverage. 
Since the disease spreads quickly, repeat sprays every 10–14 days during humid conditions. 

After harvest, plow all crop residues deep into the soil to bury the fungus and prevent carryover to the next season.""",
        "prevention": """- Plant certified disease-free seeds of resistant varieties.  
- Rotate crops with non-hosts such as corn or cotton.  
- Avoid continuous groundnut cultivation in the same field."""
    },
    
    "Early_Rust": {
        "cause": "Early signs of rust caused by *Puccinia arachidis*, favored by warm and humid weather.",
        "solution": """Scout fields regularly, especially the undersides of older leaves. Look for tiny yellow specks 
that develop into raised pustules. 

Apply fungicides such as **wettable sulfur**, **mancozeb**, or **tebuconazole** immediately at the first sign. 
Early intervention prevents spores from spreading and damaging healthy leaves.""",
        "prevention": """- Grow rust-resistant varieties.  
- Remove old crop debris and volunteer plants before sowing.  
- Maintain proper crop spacing for airflow."""
    },
    
    "Healthy": {
        "cause": "The leaf appears normal with no visible signs of disease or nutrient deficiency.",
        "solution": """No treatment is required as the plant is currently healthy. 

Maintain good crop management practices such as **balanced fertilization**, **timely irrigation**, and **weed control**.  
Regular field monitoring is recommended to catch early signs of potential problems.""",
        "prevention": """- Inspect fields weekly for early detection.  
- Maintain proper spacing for sunlight and airflow.  
- Rotate crops to prevent soil-borne issues.  
- Apply organic manure or compost to sustain soil health."""
    },
    
    "Late_Leaf_Spot": {
        "cause": "Caused by the fungus *Cercosporidium personatum*, often more severe than early leaf spot.",
        "solution": """This disease causes rapid defoliation, reducing yield drastically. 
At the first sign of nearly black spots, start spraying fungicides such as **tebuconazole** or **pyraclostrobin**. 

Spray every 10–14 days during humid conditions. Resistant varieties can greatly reduce losses.  

After harvest, plow crop debris deep into the soil to eliminate fungal survival and protect the next crop.""",
        "prevention": """- Use resistant or tolerant varieties.  
- Rotate crops for at least 2 years with non-host plants like corn.  
- Avoid planting groundnut in the same field continuously."""
    },
    
    "Nutrition_Deficiency": {
        "cause": "Caused by lack of essential nutrients like nitrogen, potassium, calcium, or micronutrients.",
        "solution": """Conduct a soil test to identify the specific nutrient deficiency. 
Apply balanced fertilizers or foliar sprays as recommended.  

For example, use gypsum for calcium deficiency or potash fertilizers for potassium shortage.""",
        "prevention": """- Test soil regularly and apply fertilizers based on results.  
- Incorporate organic compost or manure to improve soil fertility.  
- Avoid over-irrigation, which leaches nutrients away."""
    },
    
    "Rust": {
        "cause": "Caused by the fungus *Puccinia arachidis*, common in warm and humid climates.",
        "solution": """Rust appears as orange to reddish-brown pustules on the underside of leaves. 
It causes premature leaf drying and defoliation.  

Apply fungicides like **wettable sulfur**, **mancozeb**, or **tebuconazole** promptly. 
Repeat sprays every 10–14 days for effective control.  

Resistant varieties provide an additional safeguard. 
After harvest, destroy leftover debris and volunteer plants to cut fungal survival.""",
        
        "prevention": """- Plant rust-resistant groundnut varieties.  
- Keep fields free of old plant debris before new planting.  
- Ensure good airflow by avoiding overcrowded planting."""
    }
}

damage_severity = {
    "Early_Leaf_Spot": "🟠 Moderate (30–50% yield loss if untreated)",
    "Early_Rust": "🟠 Moderate (25–45% yield loss if untreated)",
    "Healthy": "🟢 No risk, plant is healthy",
    "Late_Leaf_Spot": "🔴 High (40–60% yield loss if untreated)",
    "Nutrition_Deficiency": "🟡 Moderate (20–40% yield loss if not corrected)",
    "Rust": "🔴 High (35–55% yield loss if untreated)"
}

