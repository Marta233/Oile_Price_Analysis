import React, { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, ReferenceDot, BarChart, Bar, LabelList, Legend } from 'recharts';
import './App.css';

function App() {
    const [priceData, setPriceData] = useState([]);
    const [eventData, setEventData] = useState([]);
    const [eventColors, setEventColors] = useState({});
    const [selectedEventType, setSelectedEventType] = useState('');
    const [filteredPriceData, setFilteredPriceData] = useState([]);
    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');
    const [dateFilteredData, setDateFilteredData] = useState([]);

    // Sample performance metrics data
    const modelMetrics = [
        { Model_Name: 'VAR', MSE: 0.7551, MAE: 0.6934 },
        { Model_Name: 'LSTM', MSE: 0.00016, MAE: 0.0089 },
        { Model_Name: 'ARIMA', MSE: 6.6201, MAE: 2.2773 },
        { Model_Name: 'Markov', MSE: 1.4790, MAE: 0.7497 },
    ];

    useEffect(() => {
        fetch('http://127.0.0.1:5000/api/prices')
            .then(response => response.json())
            .then(data => {
                setPriceData(data);
                setFilteredPriceData(data);
            })
            .catch(error => console.error('Error fetching price data:', error));

        fetch('http://127.0.0.1:5000/api/events')
            .then(response => response.json())
            .then(data => {
                setEventData(data);
                generateEventColors(data); // Generate colors based on event data
            })
            .catch(error => console.error('Error fetching event data:', error));
    }, []);

    const generateEventColors = (events) => {
        const colors = {};
        const uniqueEventTypes = [...new Set(events.map(event => event.event_type))];

        uniqueEventTypes.forEach((type, index) => {
            colors[type] = `hsl(${(index * 360) / uniqueEventTypes.length}, 100%, 50%)`; // Generate a unique color
        });

        setEventColors(colors);
    };

    const handleEventTypeChange = (event) => {
        const selectedType = event.target.value;
        setSelectedEventType(selectedType);

        let filteredData = [];
        if (selectedType) {
            const eventDates = eventData
                .filter(eventEntry => eventEntry.event_type.toLowerCase() === selectedType.toLowerCase())
                .map(eventEntry => eventEntry.Date);
            filteredData = priceData.filter(priceEntry => eventDates.includes(priceEntry.Date));
        } else {
            filteredData = priceData;
        }
        setFilteredPriceData(filteredData);
    };

    const handleDateFilter = () => {
        const filteredData = priceData.filter(priceEntry => {
            const entryDate = new Date(priceEntry.Date);
            const start = new Date(startDate);
            const end = new Date(endDate);
            return entryDate >= start && entryDate <= end;
        });
        setDateFilteredData(filteredData);
    };

    return (
        <div style={{ backgroundColor: 'black', minHeight: '100vh', color: 'white' }}>
            <h1 style={{ textAlign: 'center', margin: '20px 0' }}>Brent Oil Prices</h1>

            {priceData.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    {/* Main Price Line Chart with Event Highlights */}
                    <ResponsiveContainer width="70%" height={400}>
                        <LineChart data={priceData}>
                            <XAxis dataKey="Date" tick={{ angle: -45, fill: 'white' }} textAnchor="end" />
                            <YAxis tick={{ fill: 'white' }} padding={{ top: 20, bottom: 20 }} domain={['auto', 'auto']} />
                            <Tooltip />
                            <CartesianGrid strokeDasharray="3 3" stroke="gray" />

                            {/* Line for Price Data */}
                            <Line
                                type="monotone"
                                dataKey="Price"
                                stroke="#ff7300"
                                strokeWidth={2}
                                dot={false}
                                activeDot={{ r: 8 }}
                            />

                            {/* Event Highlights using ReferenceDot */}
                            {eventData.map((event, index) => (
                                <ReferenceDot
                                    key={index}
                                    x={event.Date}
                                    y={priceData.find(price => price.Date === event.Date)?.Price}
                                    r={6}
                                    fill={eventColors[event.event_type] || 'red'} // Use the dynamic color
                                    stroke="none"
                                    label={{
                                        position: 'top',
                                        value: event.event_type,
                                        fill: 'white',
                                        fontSize: 12,
                                    }}
                                />
                            ))}
                        </LineChart>
                    </ResponsiveContainer>

                    {/* Row for Event EDA - Two Columns */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', width: '70%', marginTop: '20px' }}>
                        <div style={{ backgroundColor: '#333', flex: '1', margin: '0 10px', padding: '20px', borderRadius: '8px' }}>
                            <h3 style={{ textAlign: 'center' }}>Price Over Time Based on Event Type</h3>
                            
                            {/* Dropdown for Event Type Selection */}
                            <div style={{ textAlign: 'center', marginBottom: '10px' }}>
                                <label style={{ marginRight: '10px', color: 'white' }}>Select Event Type:</label>
                                <select value={selectedEventType} onChange={handleEventTypeChange} style={{ padding: '5px', borderRadius: '5px' }}>
                                    <option value="">All Events</option>
                                    {Array.from(new Set(eventData.map(event => event.event_type))).map((eventType, index) => (
                                        <option key={index} value={eventType}>{eventType}</option>
                                    ))}
                                </select>
                            </div>

                            {/* Line chart for Price Data based on Event Type */}
                            <ResponsiveContainer width="100%" height={300}>
                                {filteredPriceData.length > 0 ? (
                                    <LineChart data={filteredPriceData}>
                                        <XAxis dataKey="Date" tick={{ angle: -45, fill: 'white' }} textAnchor="end" />
                                        <YAxis tick={{ fill: 'white' }} padding={{ top: 20, bottom: 20 }} domain={['auto', 'auto']} />
                                        <Tooltip />
                                        <CartesianGrid strokeDasharray="3 3" stroke="gray" />
                                        <Line
                                            type="monotone"
                                            dataKey="Price"
                                            stroke="#00c49f"
                                            strokeWidth={2}
                                            dot={false}
                                            activeDot={{ r: 8 }}
                                        />
                                    </LineChart>
                                ) : (
                                    <p style={{ textAlign: 'center', color: 'white' }}>No data available for the selected event type.</p>
                                )}
                            </ResponsiveContainer>
                        </div>

                        {/* Column for Date Range Filter */}
                        <div style={{ backgroundColor: '#333', flex: '1', margin: '0 10px', padding: '20px', borderRadius: '8px' }}>
                            <h3 style={{ textAlign: 'center' }}>Filter by Date Range</h3>
                            <div style={{ textAlign: 'center', marginBottom: '10px' }}>
                                <label style={{ color: 'white', marginRight: '10px' }}>Start Date:</label>
                                <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
                                <label style={{ color: 'white', margin: '0 10px' }}>End Date:</label>
                                <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
                                <button onClick={handleDateFilter} style={{ marginLeft: '10px', padding: '5px 10px' }}>Filter</button>
                            </div>

                            {/* Line chart for Price Data based on Date Range */}
                            <ResponsiveContainer width="100%" height={300}>
                                {dateFilteredData.length > 0 ? (
                                    <LineChart data={dateFilteredData}>
                                        <XAxis dataKey="Date" tick={{ angle: -45, fill: 'white' }} textAnchor="end" />
                                        <YAxis tick={{ fill: 'white' }} padding={{ top: 20, bottom: 20 }} domain={['auto', 'auto']} />
                                        <Tooltip />
                                        <CartesianGrid strokeDasharray="3 3" stroke="gray" />
                                        <Line
                                            type="monotone"
                                            dataKey="Price"
                                            stroke="#8884d8"
                                            strokeWidth={2}
                                            dot={false}
                                            activeDot={{ r: 8 }}
                                        />
                                    </LineChart>
                                ) : (
                                    <p style={{ textAlign: 'center', color: 'white' }}>No data available for the selected date range.</p>
                                )}
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* New Row for Event Highlights with Line Chart */}
                    <div style={{ backgroundColor: '#333', width: '70%', marginTop: '20px', padding: '20px', borderRadius: '8px' }}>
                        <h3 style={{ textAlign: 'center' }}>Event Highlights</h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <LineChart data={priceData}>
                                <XAxis dataKey="Date" tick={{ angle: -45, fill: 'white' }} textAnchor="end" />
                                <YAxis tick={{ fill: 'white' }} padding={{ top: 20, bottom: 20 }} domain={['auto', 'auto']} />
                                <Tooltip />
                                <CartesianGrid strokeDasharray="3 3" stroke="gray" />
                                <Line type="monotone" dataKey="Price" stroke="#ff7300" strokeWidth={2} dot={false} />

                                {/* Event Highlighters */}
                                {eventData.map((event, index) => (
                                    <Line
                                        key={index}
                                        type="monotone"
                                        dataKey="Price"
                                        stroke={eventColors[event.event_type] || '#ffffff'} // Use the dynamic color
                                        strokeWidth={2}
                                        dot={false}
                                        isAnimationActive={false}
                                        data={[
                                            { Date: event.Date, Price: 0 }, 
                                            { Date: event.Date, Price: Math.max(...priceData.map(d => d.Price)) } // Draw a vertical line
                                        ]}
                                    />
                                ))}
                            </LineChart>
                        </ResponsiveContainer>

                        {/* Legend for Event Types */}
                        <div style={{ textAlign: 'center', marginTop: '10px' }}>
                            <h4 style={{ color: 'white' }}>Event Legend</h4>
                            <div style={{ display: 'flex', justifyContent: 'center' }}>
                                {Object.entries(eventColors).map(([eventType, color]) => (
                                    <div key={eventType} style={{ display: 'flex', alignItems: 'center', margin: '0 10px' }}>
                                        <div style={{ width: '20px', height: '20px', backgroundColor: color, marginRight: '5px' }}></div>
                                        <span style={{ color: 'white' }}>{eventType}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Metrics Display Section */}
                    <div style={{ display: 'flex', justifyContent: 'center', width: '70%', marginTop: '20px' }}>
                        <div style={{ backgroundColor: '#333', flex: '1', margin: '0 10px', padding: '20px', borderRadius: '8px' }}>
                            <h3 style={{ textAlign: 'center' }}>Model Performance Metrics</h3>
                            <ResponsiveContainer width="100%" height={300}>
                                <BarChart data={modelMetrics}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="gray" />
                                    <XAxis dataKey="Model_Name" stroke="#fff" label={{ value: "Model Name", position: "bottom", offset: 0 }} />
                                    <YAxis stroke="#fff" label={{ value: "Error Metrics", angle: -90, position: 'insideLeft', offset: 0 }} />
                                    <Tooltip />
                                    <Legend />
                                    <Bar dataKey="MSE" fill="#ff7300" name="Mean Squared Error">
                                        <LabelList dataKey="MSE" position="top" fill="#fff" />
                                    </Bar>
                                    <Bar dataKey="MAE" fill="#00c49f" name="Mean Absolute Error">
                                        <LabelList dataKey="MAE" position="top" fill="#fff" />
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>
            ) : (
                <p style={{ textAlign: 'center', color: 'white' }}>Loading data...</p>
            )}
        </div>
    );
}

export default App;