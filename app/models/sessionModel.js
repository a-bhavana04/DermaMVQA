const mongoose = require('mongoose');

const sessionSchema = new mongoose.Schema({
    sessionId: { type: String, required: true, unique: true },
    email: { type: String, required: true },
    title: { type: String, default: 'Untitled Session' }
}, { timestamps: true });

sessionSchema.statics.storeSession = async function(email, sessionId) {
    await this.create({ email, sessionId });
};

sessionSchema.statics.getSessions = async function(email) {
    const sessions = await this.find({ email }).sort({ createdAt: -1 });
    return sessions;
};

sessionSchema.statics.getSessionById = async function(sessionId) {
    return await this.findOne({ sessionId });
};

const SessionModel = mongoose.model('Session', sessionSchema);
module.exports = SessionModel;
