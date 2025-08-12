const express = require('express');
const path = require('path');
const session = require('express-session');
const dotenv = require('dotenv');
const mongoose = require('mongoose');
const UserModel = require('./models/userModel');
const SessionModel = require('./models/sessionModel');
const QueryResponseModel = require('./models/queryResponseModel');
const GraphModel = require('./models/graphModel');
const multer = require('multer');
const fs = require('fs');
const upload = multer({ dest: 'uploads/' });

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// MongoDB connection
mongoose.connect(process.env.MONGO_URL || 'mongodb://localhost:27017/session_app', {
    useNewUrlParser: true,
    useUnifiedTopology: true
}).then(() => console.log('MongoDB Connected'))
  .catch(err => console.error('MongoDB Connection Error:', err));

// Middleware
app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

app.use(session({
    secret: process.env.SESSION_SECRET || 'mysecret',
    resave: false,
    saveUninitialized: false,
    cookie: { secure: false }
}));

// Serve EJS templates
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Routes
app.get('/', (req, res) => res.render('index'));
app.get('/signup', (req, res) => res.render('signup'));
app.get('/login', (req, res) => res.render('login'));

// Signup Route
app.post('/signup', async (req, res) => {
    const { username, password } = req.body;
    if (!username || !password) return res.send('Username and Password required.');

    const success = await UserModel.createUser(username, password);
    if (!success) return res.redirect('/signup'); 

    res.redirect('/login');
});

// Login Route
app.post('/login', async (req, res) => {
    const { lemail, lpassword } = req.body;

    const isAuthenticated = await UserModel.authenticateUser(lemail, lpassword);
    if (!isAuthenticated) return res.redirect('/login');

    req.session.user = lemail;
    res.redirect('/home');
});

// Home Page (Show all sessions)
app.get('/home', async (req, res) => {
    if (!req.session.user) return res.redirect('/login');

    const email = req.session.user;
    const sessions = await SessionModel.getSessions(email);

    res.render('home', { email, sessions });
});

// Create a New Session (Fixed JSON Response)
app.post('/new-session', async (req, res) => {
    if (!req.session.user) return res.status(401).json({ error: 'Unauthorized' });

    const email = req.session.user;
    const sessionId = `session:${Date.now()}`;

    await SessionModel.storeSession(email, sessionId);

    res.json({ redirect: `/session/${sessionId}` });
});

// View a Session
app.get('/session/:id', async (req, res) => {
    if (!req.session.user) return res.redirect('/login');

    const sessionId = req.params.id;
    const session = await SessionModel.getSessionById(sessionId);
    const queryResponses = await QueryResponseModel.find({ sessionId }).sort({ order: 1 });

    if (!session) return res.redirect('/home');

    res.render('session', { 
        sessionId, 
        sessionTitle: session.title || 'Untitled Session', 
        queryResponses 
    });
});

// Add Query to Session
const axios = require('axios');

app.post('/session/:id/add-query', upload.single('image'), async (req, res) => {
    if (!req.session.user) return res.redirect('/login');

    const sessionId = req.params.id;
    const { query } = req.body;

    const lastEntry = await QueryResponseModel.findOne({ sessionId }).sort({ order: -1 });
    const newOrder = lastEntry ? lastEntry.order + 1 : 1;

    let base64Image = null;
    if (req.file) {
        const imageBuffer = fs.readFileSync(req.file.path);
        base64Image = imageBuffer.toString('base64');
        fs.unlinkSync(req.file.path); // clean up after reading
    }

    try {
        const response = await axios.post('http://localhost:5002/query', { 
            query, 
            image: base64Image 
        });
        const answer = response.data.answer;

        await QueryResponseModel.create({ sessionId, query, response: answer, order: newOrder });
    } catch (error) {
        console.error("Error calling Python API:", error);
        await QueryResponseModel.create({ sessionId, query, response: "Error fetching response from AI.", order: newOrder });
    }

    res.redirect(`/session/${sessionId}`);
});

// Logout Route
app.get('/logout', (req, res) => {
    req.session.destroy(() => res.redirect('/'));
});

// Start Server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
