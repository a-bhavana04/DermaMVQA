const mongoose = require('mongoose');

const graphDataSchema = new mongoose.Schema({
    sessionId: { type: String, required: true },
    node1: { type: String, required: true },
    node2: { type: String, required: true },
    edge: { type: String, required: true }
}, { timestamps: true });

graphDataSchema.statics.storeGraphData = async function(sessionId, node1, node2, edge) {
    await this.create({ sessionId, node1, node2, edge });
};

graphDataSchema.statics.getGraphData = async function(sessionId) {
    return await this.find({ sessionId });
};

const GraphModel = mongoose.model('GraphData', graphDataSchema);
module.exports = GraphModel;
