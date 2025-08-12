const mongoose = require('mongoose');

const QueryResponseSchema = new mongoose.Schema({
    sessionId: { 
        type: String, 
        ref: 'Session', 
        required: true 
    },
    query: { 
        type: String, 
        required: true 
    },
    response: { 
        type: String, 
        required: true 
    },
    order: { 
        type: Number, 
        required: true 
    }
}, { timestamps: true });

const QueryResponseModel = mongoose.model('QueryResponse', QueryResponseSchema);
module.exports = QueryResponseModel;
