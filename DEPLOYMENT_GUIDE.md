# 🎮 Interactive LoL Match Predictor - Deployment Guide

## 🚀 Quick Start for Real Matches

### **Step 1: Run the Interactive Predictor**
```bash
cd Tese
python interactive_match_predictor.py
```

### **Step 2: Choose Your Mode**
- **Mode 1**: Single Game Prediction
- **Mode 2**: Best-of Series (BO3/BO5) - **RECOMMENDED for KT vs GenG**

---

## 🏆 Example: KT vs GenG LCK Series

### **Team Selection**
```
Blue Side Team:
   League: LCK
   Team: KT

Red Side Team:  
   League: LCK
   Team: GENG
```

### **🎯 PROFESSIONAL DRAFT FORMAT**
The script now follows the **exact professional League of Legends draft sequence**:

#### **🚫 First Ban Phase (6 bans)**
```
1. 🔴 GenG Ban 1    →  Azir
2. 🔵 KT Ban 1      →  Yone  
3. 🔵 KT Ban 2      →  Viego
4. 🔴 GenG Ban 2    →  LeBlanc
5. 🔵 KT Ban 3      →  Kai'Sa
6. 🔴 GenG Ban 3    →  Graves
```

#### **🎯 First Pick Phase (6 picks)**
```
1. 🔵 KT Pick 1     →  Jax (flex)
2. 🔴 GenG Pick 1   →  Xin Zhao (jungle)
3. 🔴 GenG Pick 2   →  Ahri (mid)  
4. 🔵 KT Pick 2     →  Varus (bot)
5. 🔵 KT Pick 3     →  Thresh (support)
6. 🔴 GenG Pick 3   →  Ornn (top)
```

#### **🚫 Second Ban Phase (4 bans)**
```
1. 🔴 GenG Ban 4    →  Jinx
2. 🔵 KT Ban 4      →  Nautilus
3. 🔴 GenG Ban 5    →  Elise
4. 🔵 KT Ban 5      →  Jhin
```

#### **🎯 Final Pick Phase (4 picks)**
```
1. 🔴 GenG Pick 4   →  Kalista (bot)
2. 🔵 KT Pick 4     →  Nidalee (jungle)
3. 🔵 KT Pick 5     →  Yasuo (mid → Jax goes top)
4. 🔴 GenG Pick 5   →  Leona (support)
```

#### **🎭 Role Assignment**
After picks, you assign champions to their actual roles:
```
🔵 KT Final Comp:
   Top: Jax
   Jungle: Nidalee  
   Mid: Yasuo
   Bot: Varus
   Support: Thresh

🔴 GenG Final Comp:
   Top: Ornn
   Jungle: Xin Zhao
   Mid: Ahri
   Bot: Kalista
   Support: Leona
```

---

## 🎯 Features

### **✅ Professional Draft Simulation**
- **Exact professional pick/ban order**
- **Real-time draft tracking**
- **Flex pick handling**
- **Role assignment after draft**

### **✅ Interactive Input**
- Team selection from major leagues (LCK, LEC, LCS, LPL)
- Champion name validation (handles "Xin Zhao", "Kai'Sa", etc.)
- Draft phase indicators and guidance
- Error handling and input validation

### **✅ Dual-Perspective Prediction**
- Predicts from both Blue and Red side perspectives
- Averages probabilities for balanced prediction
- Confidence intervals and uncertainty analysis

### **✅ Best-of Series Support**
- BO3 or BO5 format
- Game-by-game predictions with full draft
- Running score tracking
- Series winner determination

### **✅ Real-World Ready**
- Uses your trained 82.97% AUC model
- All 33 advanced features with professional encoding
- Handles missing data gracefully
- Dual-perspective prediction for balance

---

## 📊 Expected Output

### **Draft Summary**
```
📊 DRAFT SUMMARY
🔵 KT:
   Top: Jax
   Jungle: Nidalee
   Mid: Yasuo
   Bot: Varus
   Support: Thresh
   Bans: Yone, Viego, Kai'Sa, Nautilus, Elise

🔴 GENG:
   Top: Ornn
   Jungle: Xin Zhao
   Mid: Ahri
   Bot: Kalista  
   Support: Leona
   Bans: Azir, LeBlanc, Graves, Jinx, Jhin
```

### **Prediction Result**
```
🏆 PREDICTION RESULTS:
   🔵 KT: 45.2% win probability
   🔴 GENG: 54.8% win probability

🎯 PREDICTED WINNER: GENG
   📊 Confidence: 54.8%
   ⚠️ Moderate confidence prediction
```

---

## 🔧 Technical Details

### **Professional Draft Accuracy**
- **Exact Riot Games draft sequence**
- **Side selection matters** (Blue gets first pick advantage)
- **Ban timing reflects meta importance**
- **Flex picks handled realistically**

### **Champion Name Formatting**
- **Spaces preserved**: "Xin Zhao" not "XinZhao"
- **Apostrophes preserved**: "Kai'Sa" not "Kaisa"  
- **Case insensitive**: "xin zhao" → "Xin Zhao"
- **Validation included**: Suggests similar names for typos

### **Model Integration**
- Uses your trained 82.97% AUC model
- All 33 advanced features with professional encoding
- Handles missing data gracefully
- Dual-perspective prediction for balance

### **Feature Processing**
- 33 advanced features per match
- Champion characteristics and meta strength
- Team composition synergy
- Role-specific encoding
- Temporal and patch adjustments

### **Prediction Method**
- Dual-perspective modeling (Blue + Red side)
- Probability averaging for balanced results
- Confidence scoring based on prediction strength
- Real-time feature generation

---

## 🎯 Best Practices for Real Predictions

### **1. Accurate Draft Input**
- **Follow actual draft order** from live games
- **Use exact champion names** (script validates)
- **Assign roles correctly** after flex picks
- **Include all bans** even if obvious

### **2. Series Prediction**
- **Use BO3 for regular season** (like LCK Spring)
- **Use BO5 for playoffs/finals**
- **Track actual draft progression**
- **Compare predictions with real results**

### **3. Real-World Usage**
- **Predict before game starts** for best accuracy
- **Use during draft phase** for strategic insights
- **Track confidence levels** vs actual outcomes
- **Note meta shifts** between patches

---

## 🚀 Usage Examples

### **For Live LCK Games:**
```python
# Watch LCK live, input draft as it happens
python interactive_match_predictor.py

# Select BO3 series mode
# Input: KT vs GenG  
# Follow exact draft sequence
# Get real-time predictions
```

### **For Champion Names:**
```python
# Use champion name guide if unsure
python champion_name_guide.py

# Test: "xin zhao" → Valid ✅
# Test: "kaisa" → Suggests "Kai'Sa" 
```

### **For Fantasy/Betting:**
```python
# Multiple daily matches
# Single game mode for quick predictions
# Confidence-based decision making
```

---

## ✅ Ready to Predict!

**New Professional Features:**
- 🎯 **Real draft simulation** - exactly like pro games
- 🔧 **Role assignment handling** - deals with flex picks  
- 📝 **Champion validation** - handles special names
- 🎮 **Live game ready** - input as draft happens

**Perfect for:**
- 📺 Live LCK/LEC/LCS predictions during broadcast
- 🏆 Series outcome prediction with full draft analysis
- 💰 Fantasy league and betting with professional accuracy
- 📊 Real-world model validation on current meta

**Start with:** KT vs GenG series using the exact professional draft format! 🚀 