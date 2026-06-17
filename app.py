from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

app = Flask(__name__, static_folder='static')

# Load dataset
DATA_PATH = 'Crop_recommendationV2.csv'
df = pd.read_csv(DATA_PATH)

# Prepare features and target
DROP_COLS = ['label', 'co2_concentration', 'crop_density', 'pest_pressure', 'frost_risk']
X = df.drop(DROP_COLS, axis=1)
y = df['label']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Calculate model accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
model_accuracy = accuracy_score(y_test, y_pred)

CROP_IMAGES = {
    'rice': 'https://media.istockphoto.com/id/153737841/photo/rice.jpg?s=612x612&w=0&k=20&c=lfO7iLT0UsDDzra0uBOsN1rvr2d5OEtrG2uwbts33_c=',
    'wheat': 'https://images.unsplash.com/photo-1437252611977-07f74518abd7?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8d2hlYXQlMjBoYXJ2ZXN0fGVufDB8fDB8fHww',
    'maize': 'https://t3.ftcdn.net/jpg/05/43/33/90/360_F_543339060_47tywFrfEyj9NdrZEw0DVxDSnBPHw5Jn.jpg',
    'cotton': 'https://cdn.britannica.com/72/270772-050-9B03FF80/Cotton-plants-in-a-field.jpg',
    'sugarcane': 'https://sasmabv.com/wp-content/uploads/2021/07/sugarcane-alcohol.jpg',
    'jute': 'https://m.media-amazon.com/images/I/71Ht1HNceXL._UF894,1000_QL80_.jpg',
    'barley': 'https://www.tastingtable.com/img/gallery/everything-you-need-to-know-about-barley/l-intro-1640793289.jpg',
    'millet': 'https://agritimes.co.in/assets/images/millets-smart-grain-powering-sustainable-agriculture-english.jpeg',
    'soybean': 'https://images.unsplash.com/photo-1502741338009-cac2772e18bc?auto=format&fit=crop&w=400&q=80',
    'groundnut': 'https://www.shutterstock.com/image-photo/farmer-s-hand-cradling-freshly-260nw-2512774003.jpg',
    'pulses': 'https://www.foodunfolded.com/media/images/Africa_Pulses_header.webp',
    'potato': 'https://static.vecteezy.com/system/resources/thumbnails/058/178/528/small_2x/fresh-potatoes-growing-in-a-fertile-field-under-sunlight-photo.jpg',
    'onion': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSxe1RKIlg45XDtbbNJTBUxBSsOWGgl8cI3h5R2q2FAu0d7EZcRTDs7LGYawMhxPKrm2-Y&usqp=CAU',
    'tomato': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTl7sATwiTU1Mn6QOsgEQGy0jJpqwqJlB2jUA&s',
    'banana': 'https://m.media-amazon.com/images/I/51MH9IBAGHS._UF1000,1000_QL80_.jpg',
    'mango': 'https://www.fortheloveofnature.in/cdn/shop/products/Mangiferaindica-Priyur_Mango_1.jpg?v=1640246617',
    'muskmelon': 'https://www.greendna.in/cdn/shop/products/Muskmelon-cover-1.jpg?v=1746694301',
    'mothbeans': 'https://www.aranyapurefood.com/cdn/shop/files/WhatsAppImage2024-03-27at10.19.02AM.jpg?v=1711515146',
    'coffee': 'https://media.istockphoto.com/id/489377142/photo/white-cup-with-coffee-beans-on-dark-background.jpg?s=612x612&w=0&k=20&c=UZvTyckcO1yCxS4Tjd9yF208djXfrbSA-7vsad7CRUE=',
    'pigeonpeas': 'https://i.cdn.newsbytesapp.com/images/l87220250219112949.jpeg',
    'grapes': 'https://images.pexels.com/photos/708777/pexels-photo-708777.jpeg',
    'papaya': 'https://www.shutterstock.com/image-photo/big-orange-papaya-gree-tree-600nw-2489304241.jpg',
    'apple': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQSG7CcTc-_eP7eW8rMi7hA1eAuP57__AGMEQ&s',
    'peaches': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS05KC1V84s47O1GViVFXfVsY-ZQAb_VYgE8w&s',
    'cherry': 'https://www.saharanpurnursery.in/cdn/shop/files/WhatsAppImage2024-03-11at18.55.37.jpg?v=1718609003',
    'rose': 'https://bouqs.com/blog/wp-content/uploads/2018/08/shutterstock_1662182848-min.jpg',
    'sunflower': 'https://rukminim2.flixcart.com/image/850/1000/kpmy8i80/plant-seed/s/y/v/30-sunflower-flower-tall-sungold-30-seeds-with-100-natural-cow-original-imag3t4zpwzcxqg4.jpeg?q=90&crop=false',
    'kidneybeans': 'https://m.media-amazon.com/images/I/919VDk6WrbL._UF1000,1000_QL80_.jpg',
    'chickpea': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRLrfzXlI18998wKGfpStgSBmY3fcVVOUFFUg&s',
    # Add more crops and images as needed
}

# Crop descriptions and information
CROP_DESCRIPTIONS = {
    'rice': {
        'description': 'Rice is a staple food crop and the most important cereal crop in India. It is grown in tropical and subtropical regions with high rainfall.',
        'climate': 'Warm and humid climate with temperature 20-35°C',
        'soil': 'Clay loam soil with good water retention',
        'water_requirement': 'High water requirement, needs standing water',
        'season': 'Kharif (June-October) and Rabi (November-March)',
        'nutritional_value': 'Rich in carbohydrates, provides energy',
        'economic_importance': 'Major export crop, supports millions of farmers'
    },
    'wheat': {
        'description': 'Wheat is the second most important cereal crop in India. It is a winter crop that requires cool weather during growth and warm weather during ripening.',
        'climate': 'Cool climate during growing season, 15-25°C',
        'soil': 'Well-drained loamy soil',
        'water_requirement': 'Moderate water requirement',
        'season': 'Rabi season (October-March)',
        'nutritional_value': 'Rich in protein, fiber, and minerals',
        'economic_importance': 'Major food grain, essential for food security'
    },
    'maize': {
        'description': 'Maize (corn) is a versatile crop used for food, feed, and industrial purposes. It is the third most important cereal crop in India.',
        'climate': 'Warm climate, temperature 18-32°C',
        'soil': 'Well-drained fertile soil',
        'water_requirement': 'Moderate water requirement',
        'season': 'Kharif and Rabi seasons',
        'nutritional_value': 'Good source of carbohydrates and protein',
        'economic_importance': 'Used in food industry, animal feed, and biofuels'
    },
    'cotton': {
        'description': 'Cotton is known as the "White Gold" of India. It is a major cash crop and the backbone of the textile industry.',
        'climate': 'Warm climate, temperature 20-35°C',
        'soil': 'Black cotton soil or alluvial soil',
        'water_requirement': 'Moderate water requirement',
        'season': 'Kharif season (June-October)',
        'nutritional_value': 'Fiber crop, not consumed directly',
        'economic_importance': 'Major export crop, supports textile industry'
    },
    'sugarcane': {
        'description': 'Sugarcane is a major commercial crop and the primary source of sugar in India. It is also used for ethanol production.',
        'climate': 'Tropical climate, temperature 20-38°C',
        'soil': 'Deep, well-drained fertile soil',
        'water_requirement': 'High water requirement',
        'season': 'Year-round cultivation possible',
        'nutritional_value': 'Source of sugar and energy',
        'economic_importance': 'Major sugar industry, ethanol production'
    },
    'jute': {
        'description': 'Jute is known as the "Golden Fiber" and is used for making sacks, bags, and other packaging materials.',
        'climate': 'Warm and humid climate',
        'soil': 'Alluvial soil with good drainage',
        'water_requirement': 'High water requirement',
        'season': 'Kharif season',
        'nutritional_value': 'Fiber crop, not consumed directly',
        'economic_importance': 'Eco-friendly packaging material'
    },
    'barley': {
        'description': 'Barley is a hardy cereal crop that can grow in adverse conditions. It is used for food, feed, and brewing.',
        'climate': 'Cool climate, temperature 12-25°C',
        'soil': 'Well-drained loamy soil',
        'water_requirement': 'Low to moderate water requirement',
        'season': 'Rabi season',
        'nutritional_value': 'Rich in fiber and minerals',
        'economic_importance': 'Used in brewing industry and animal feed'
    },
    'millet': {
        'description': 'Millets are small-seeded grasses that are highly nutritious and drought-resistant. They are considered superfoods.',
        'climate': 'Warm climate, drought-tolerant',
        'soil': 'Well-drained soil',
        'water_requirement': 'Low water requirement',
        'season': 'Kharif season',
        'nutritional_value': 'High in protein, fiber, and minerals',
        'economic_importance': 'Climate-smart crop, food security'
    },
    'soybean': {
        'description': 'Soybean is a leguminous crop rich in protein and oil. It is used for food, feed, and industrial purposes.',
        'climate': 'Warm climate, temperature 20-30°C',
        'soil': 'Well-drained fertile soil',
        'water_requirement': 'Moderate water requirement',
        'season': 'Kharif season',
        'nutritional_value': 'Excellent source of protein and oil',
        'economic_importance': 'Major oilseed crop, protein source'
    },
    'groundnut': {
        'description': 'Groundnut (peanut) is an important oilseed crop. It is grown for its edible seeds and oil.',
        'climate': 'Warm climate, temperature 20-30°C',
        'soil': 'Sandy loam soil',
        'water_requirement': 'Moderate water requirement',
        'season': 'Kharif and Rabi seasons',
        'nutritional_value': 'Rich in protein, oil, and minerals',
        'economic_importance': 'Major oilseed crop, edible oil source'
    },
    'pulses': {
        'description': 'Pulses include various legumes like chickpea, pigeon pea, and lentils. They are rich in protein.',
        'climate': 'Cool to warm climate',
        'soil': 'Well-drained soil',
        'water_requirement': 'Low to moderate water requirement',
        'season': 'Rabi season',
        'nutritional_value': 'Excellent source of protein',
        'economic_importance': 'Protein security, soil fertility'
    },
    'potato': {
        'description': 'Potato is a major vegetable crop and the fourth most important food crop globally.',
        'climate': 'Cool climate, temperature 15-25°C',
        'soil': 'Loose, well-drained soil',
        'water_requirement': 'Moderate water requirement',
        'season': 'Rabi season',
        'nutritional_value': 'Good source of carbohydrates and vitamin C',
        'economic_importance': 'Major vegetable crop, food security'
    },
    'onion': {
        'description': 'Onion is an essential vegetable crop used in almost every Indian household for cooking.',
        'climate': 'Moderate climate',
        'soil': 'Well-drained fertile soil',
        'water_requirement': 'Moderate water requirement',
        'season': 'Rabi season',
        'nutritional_value': 'Contains antioxidants and vitamins',
        'economic_importance': 'Essential vegetable, high demand'
    },
    'tomato': {
        'description': 'Tomato is a versatile vegetable crop used in various cuisines and processed products.',
        'climate': 'Warm climate, temperature 20-30°C',
        'soil': 'Well-drained fertile soil',
        'water_requirement': 'Moderate water requirement',
        'season': 'Year-round cultivation',
        'nutritional_value': 'Rich in vitamins A and C, lycopene',
        'economic_importance': 'Major vegetable crop, processing industry'
    },
    'banana': {
        'description': 'Banana is a major fruit crop and one of the most consumed fruits globally.',
        'climate': 'Tropical climate, temperature 20-35°C',
        'soil': 'Rich, well-drained soil',
        'water_requirement': 'High water requirement',
        'season': 'Year-round cultivation',
        'nutritional_value': 'Rich in potassium and vitamins',
        'economic_importance': 'Major fruit crop, export potential'
    },
    'mango': {
        'description': 'Mango is known as the "King of Fruits" and is India\'s national fruit.',
        'climate': 'Tropical climate, temperature 20-35°C',
        'soil': 'Deep, well-drained soil',
        'water_requirement': 'Moderate water requirement',
        'season': 'Summer season',
        'nutritional_value': 'Rich in vitamins A and C',
        'economic_importance': 'Major fruit crop, export commodity'
    },
    'muskmelon': {
        'description': 'Muskmelon is a refreshing summer fruit with high water content and sweet taste.',
        'climate': 'Warm climate, temperature 20-35°C',
        'soil': 'Well-drained sandy loam soil',
        'water_requirement': 'Moderate water requirement',
        'season': 'Summer season',
        'nutritional_value': 'High water content, vitamins A and C',
        'economic_importance': 'Summer fruit crop, local markets'
    },
    'mothbeans': {
        'description': 'Mothbeans are drought-resistant legumes rich in protein and commonly grown in arid regions.',
        'climate': 'Hot and dry climate, drought-tolerant',
        'soil': 'Sandy soil, well-drained',
        'water_requirement': 'Low water requirement',
        'season': 'Kharif season',
        'nutritional_value': 'High protein content, essential nutrients',
        'economic_importance': 'Drought-resistant crop, protein source'
    },
    'coffee': {
        'description': 'Coffee is a tropical evergreen shrub that produces coffee beans, one of the most traded agricultural commodities globally.',
        'climate': 'Tropical climate, temperature 15-25°C, high humidity',
        'soil': 'Well-drained, fertile soil rich in organic matter',
        'water_requirement': 'Moderate to high water requirement',
        'season': 'Year-round cultivation, harvest 8-9 months after flowering',
        'nutritional_value': 'Contains caffeine, antioxidants, and essential minerals',
        'economic_importance': 'Major export crop, supports millions of farmers worldwide'
    },
    'pigeonpeas': {
        'description': 'Pigeon peas (arhar/toor dal) are an important pulse crop in India, known for their high protein content and drought tolerance.',
        'climate': 'Warm climate, temperature 20-35°C, drought-tolerant',
        'soil': 'Well-drained sandy loam to clay loam soil',
        'water_requirement': 'Low to moderate water requirement',
        'season': 'Kharif season (June-October)',
        'nutritional_value': 'Excellent source of protein, fiber, and essential minerals',
        'economic_importance': 'Major pulse crop, protein security, soil fertility improvement'
    },
    'grapes': {
        'description': 'Grapes are a versatile fruit crop grown for fresh consumption, wine production, and dried fruits (raisins).',
        'climate': 'Temperate to subtropical climate, temperature 15-35°C',
        'soil': 'Well-drained loamy soil with good organic matter',
        'water_requirement': 'Moderate water requirement, needs regular irrigation',
        'season': 'Summer to early autumn harvest',
        'nutritional_value': 'Rich in antioxidants, vitamins C and K, and resveratrol',
        'economic_importance': 'Major fruit crop, wine industry, export commodity'
    },
    'papaya': {
        'description': 'Papaya is a tropical fruit known for its sweet taste and numerous health benefits. It is rich in enzymes and vitamins.',
        'climate': 'Tropical climate, temperature 20-35°C, frost-sensitive',
        'soil': 'Well-drained sandy loam to clay loam soil, pH 6.0-6.5',
        'water_requirement': 'Moderate to high water requirement, needs regular irrigation',
        'season': 'Year-round cultivation in tropical regions',
        'nutritional_value': 'Excellent source of vitamin C, vitamin A, folate, and papain enzyme',
        'economic_importance': 'Important tropical fruit crop, export potential, medicinal uses'
    },
    'apple': {
        'description': 'Apple is one of the most popular and widely consumed fruits globally, known for its crisp texture and sweet-tart flavor.',
        'climate': 'Temperate climate, temperature 15-25°C, requires winter chilling',
        'soil': 'Well-drained loamy soil, pH 6.0-7.0, rich in organic matter',
        'water_requirement': 'Moderate water requirement, needs regular irrigation during growing season',
        'season': 'Autumn harvest (September-November in Northern Hemisphere)',
        'nutritional_value': 'Rich in fiber, vitamin C, antioxidants, and polyphenols',
        'economic_importance': 'Major fruit crop worldwide, extensive processing industry, export commodity'
    },
    'peaches': {
        'description': 'Peaches are delicious stone fruits known for their juicy, sweet flesh and fuzzy skin. They are popular in both fresh and processed forms.',
        'climate': 'Temperate to subtropical climate, temperature 15-30°C, requires winter chilling',
        'soil': 'Well-drained sandy loam to clay loam soil, pH 6.0-7.0',
        'water_requirement': 'Moderate to high water requirement, needs regular irrigation',
        'season': 'Summer harvest (June-August in Northern Hemisphere)',
        'nutritional_value': 'Rich in vitamins A and C, potassium, fiber, and antioxidants',
        'economic_importance': 'Important stone fruit crop, fresh market and processing industry'
    },
    'cherry': {
        'description': 'Cherries are small, round stone fruits known for their bright red color and sweet-tart flavor. They are popular for fresh consumption and processing.',
        'climate': 'Temperate climate, temperature 10-25°C, requires winter chilling period',
        'soil': 'Well-drained loamy soil, pH 6.0-7.5, rich in organic matter',
        'water_requirement': 'Moderate water requirement, needs regular irrigation during growing season',
        'season': 'Late spring to early summer harvest (May-July in Northern Hemisphere)',
        'nutritional_value': 'Rich in antioxidants, vitamin C, potassium, and anthocyanins',
        'economic_importance': 'High-value fruit crop, fresh market, processing for jams and juices'
    },
    'rose': {
        'description': 'Roses are ornamental flowering plants known for their beautiful blooms, fragrance, and cultural significance. They are widely used in landscaping and floriculture.',
        'climate': 'Temperate to subtropical climate, temperature 15-30°C, some varieties frost-tolerant',
        'soil': 'Well-drained loamy soil, pH 6.0-7.0, rich in organic matter',
        'water_requirement': 'Moderate water requirement, needs regular watering during growing season',
        'season': 'Spring to fall blooming (March-October), varies by variety and climate',
        'nutritional_value': 'Rose petals contain antioxidants, vitamin C, and essential oils',
        'economic_importance': 'Major ornamental crop, floriculture industry, essential oil production, landscaping'
    },
    'sunflower': {
        'description': 'Sunflowers are tall, bright yellow flowering plants known for their large flower heads that follow the sun. They are grown for oil, seeds, and ornamental purposes.',
        'climate': 'Warm temperate to subtropical climate, temperature 18-35°C, drought-tolerant',
        'soil': 'Well-drained fertile soil, pH 6.0-7.5, can grow in various soil types',
        'water_requirement': 'Moderate water requirement, drought-resistant once established',
        'season': 'Summer to early autumn (June-October), 80-120 days to maturity',
        'nutritional_value': 'Seeds rich in vitamin E, healthy fats, protein, and minerals',
        'economic_importance': 'Major oilseed crop, edible seeds, ornamental plant, biodiesel production'
    },
    'kidneybeans': {
        'description': 'Kidney beans are nutritious legumes known for their kidney-shaped seeds and high protein content. They are a staple food in many cuisines worldwide.',
        'climate': 'Warm temperate to tropical climate, temperature 20-30°C, frost-sensitive',
        'soil': 'Well-drained loamy soil, pH 6.0-7.0, rich in organic matter',
        'water_requirement': 'Moderate water requirement, needs regular irrigation',
        'season': 'Spring to summer planting, 90-120 days to maturity',
        'nutritional_value': 'Excellent source of protein, fiber, iron, folate, and antioxidants',
        'economic_importance': 'Important pulse crop, protein source, food security, soil fertility improvement'
    },
    'chickpea': {
        'description': 'Chickpea (garbanzo bean) is a major pulse crop valued for its high protein content and versatility in cooking. It is a staple in many vegetarian diets worldwide.',
        'climate': 'Cool to warm climate, temperature 15-30°C, drought-tolerant',
        'soil': 'Well-drained sandy loam to clay loam soil, pH 6.0-7.0',
        'water_requirement': 'Low to moderate water requirement, drought-resistant',
        'season': 'Winter to spring planting, 90-120 days to maturity',
        'nutritional_value': 'Excellent source of protein, fiber, iron, folate, and complex carbohydrates',
        'economic_importance': 'Major pulse crop, protein source, food security, soil fertility improvement'
    }
}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', columns=list(X.columns))

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(request.form[col]) for col in X.columns]
    
    # Get prediction and confidence scores
    pred = model.predict([data])[0]
    confidence_scores = model.predict_proba([data])[0]
    max_confidence = max(confidence_scores)
    
    # Get crop information
    crop_info = CROP_DESCRIPTIONS.get(pred, {})
    image_url = CROP_IMAGES.get(pred, '')
    
    return jsonify({
        'prediction': pred,
        'image_url': image_url,
        'accuracy': round(model_accuracy * 100, 2),
        'confidence': round(max_confidence * 100, 2),
        'description': crop_info.get('description', ''),
        'climate': crop_info.get('climate', ''),
        'soil': crop_info.get('soil', ''),
        'water_requirement': crop_info.get('water_requirement', ''),
        'season': crop_info.get('season', ''),
        'nutritional_value': crop_info.get('nutritional_value', ''),
        'economic_importance': crop_info.get('economic_importance', '')
    })

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET'])
def signup():
    return render_template('signup.html')

@app.route('/signin', methods=['GET'])
def signin():
    return render_template('signin.html')

@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('images/favicon.svg')

if __name__ == '__main__':
    app.run(debug=True) 