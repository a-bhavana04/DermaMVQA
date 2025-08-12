const mongoose = require('mongoose');
const bcrypt = require('bcrypt');

const userSchema = new mongoose.Schema({
    username: { type: String, unique: true, required: true },
    password: { type: String, required: true }
});

userSchema.statics.createUser = async function(username, password) {
    const existingUser = await this.findOne({ username });
    if (existingUser) return false;

    const hashedPassword = await bcrypt.hash(password, 10);
    await this.create({ username, password: hashedPassword });
    return true;
};

userSchema.statics.authenticateUser = async function(username, password) {
    const user = await this.findOne({ username });
    if (!user) return false;

    return await bcrypt.compare(password, user.password);
};

const UserModel = mongoose.model('User', userSchema);
module.exports = UserModel;
