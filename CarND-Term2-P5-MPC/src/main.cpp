#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "json.hpp"
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "poly_utils.h"

#include "cxxopts.hpp"

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

int main(int argc, char* argv[]) {
  // load reference speed
  double ref_v   = 70.0;
  double lower_v = 40.0;
  bool   verbose = false;
  try {
    cxxopts::Options options(argv[0], " - MPC");
    options.positional_help("[-v VELOCITY]");
    options.add_options()
      ("v,verbose",    "A flag turn on/off debug", cxxopts::value<bool>(verbose))
      ("s,speed",      "A reference speed MPH",    cxxopts::value<double>(), "Reference velocity")
      ("l,lower_speed","A lower speed MPH",        cxxopts::value<double>(), "Lower velocity");

    options.parse(argc, argv);

    // if we want to overwrite reference speed
    if (options.count("s")) {
      ref_v = options["s"].as<double>();
    }

    if (options.count("l")) {
      lower_v = options["l"].as<double>();
    }
  } catch (const cxxopts::OptionException& e) {
    cout << "error parsing options: " << e.what() << endl;
    exit(1);
  }
  // ensure lower_v < ref_v and < 50.
  lower_v = min(50., min(lower_v, ref_v * 0.8));
  cout << "MPC start with reference speed = " << ref_v << " and lower speed = " << lower_v << endl;

  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc(ref_v, lower_v, verbose);

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        if (mpc.IsVerbose()) {
          cout << "js-data " << s << endl;
        }

        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px  = j[1]["x"];
          double py  = j[1]["y"];
          double psi = j[1]["psi"];
          double v   = j[1]["speed"];
          
          /*
           * convert to local-car coordinate
           */
          double cospsi   = cos(psi);
          double sinpsi   = sin(psi);
          size_t n_points = ptsx.size();

          Eigen::VectorXd xvals(n_points);
          Eigen::VectorXd yvals(n_points);
          for(size_t i = 0; i < n_points; ++i) {
            double dx = ptsx[i] - px;
            double dy = ptsy[i] - py;
            xvals[i] =  dx * cospsi + dy * sinpsi;
            yvals[i] = -dx * sinpsi + dy * cospsi;
          }
          auto coeffs = Utils::polyfit(xvals, yvals, 3);
          // compute cte, epsi
          // note that in local-car coordinate we have (px, py, psi) -> (0, 0, 0)
          double cte  = coeffs[0];         // polyeval(coeffs, px) - py         | px = py = 0  
          double epsi = -atan(coeffs[1]);  // psi - atan(polyeval'(coeffs, px)) | px = psi = 0 

          Eigen::VectorXd state(6);

          state << 0, 0, 0, v, cte, epsi;

          /*
          * TODO: Calculate steering angle and throttle using MPC.
          *
          * Both are in between [-1, 1].
          *
          */
          auto acctuators = mpc.Solve(state, coeffs);

          double steer_value    = acctuators[0];
          double throttle_value = acctuators[1];

          // update previous acctuators (which is used for next step due to latency)
          mpc.Update(acctuators);

          json msgJson;
          // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
          // Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].
          msgJson["steering_angle"] = steer_value / 0.436332;
          msgJson["throttle"]       = throttle_value;

          //Display the MPC predicted trajectory 
          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Green line

          msgJson["mpc_x"] = mpc.mpc_x;
          msgJson["mpc_y"] = mpc.mpc_y;

          //Display the waypoints/reference line
          vector<double> next_x_vals;
          vector<double> next_y_vals;
          for(size_t i = 0; i < n_points; ++i){
            next_x_vals.push_back(xvals[i]);
            next_y_vals.push_back(yvals[i]);
          }


          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Yellow line

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;


          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          // std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    // ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
