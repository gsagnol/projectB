var express = require('express');
var path = require('path');
var solutions = require('./solutions.js');
var app = express();


app.get('/solutions', function (req, response) {
    solutions.getNames(function(solutions){
        response.send(solutions);
    });
});

app.get('/solutions/:solution/trips', function (req, response) {
    if (req.query !== {}) {
        if(req.query["maxLon"] === undefined) {
            if(req.query["quality"] == 'low'){
                solutions.getWorseTrips(req.params.solution, function(trips) {
                    response.send(trips);
                });
            }
        }else{
            var area = {
                maxLon: req.query["maxLon"],
                maxLat: req.query["maxLat"],
                minLon: req.query["minLon"],
                minLat: req.query["minLat"]
            };
            solutions.getTripsInArea(req.params.solution, area, function (trips) {
                response.send(trips);
            });
        }
    }else{
        solutions.getTrips(req.params.solution, function (trips) {
            response.send(trips);
        });
    }
});

app.get('/solutions/:solution/trips/:trip/', function (req, response) {
    solutions.getTrip(req.params.solution, req.params.trip, function(trip){
        response.send(trip);
    });
});

app.use(express.static(path.join(__dirname, 'static')));

var server = app.listen(8081, function () {

    var host = server.address().address;
    var port = server.address().port;

    console.log("Example app listening at http://%s:%s", host, port)

});