#include <uWS/uWS.h>
#include <iostream>
#include "json.hpp"
#include <math.h>
#include "ukf.h"

using namespace std;

// for convenience
using json = nlohmann::json;

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
std::string hasData(std::string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("]");
  if (found_null != std::string::npos) {
    return "";
  }
  else if (b1 != std::string::npos && b2 != std::string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

int main()
{
  uWS::Hub h;

  // Create a UKF instance
  UKF ukf;
  
  double target_x = 0.0;
  double target_y = 0.0;

  h.onMessage([&ukf,&target_x,&target_y](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event

    if (length && length > 2 && data[0] == '4' && data[1] == '2')
    {

      auto s = hasData(std::string(data));
      if (s != "") {
      	
      	
        auto j = json::parse(s);
        std::string event = j[0].get<std::string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object

          double hunter_x = std::stod(j[1]["hunter_x"].get<std::string>());
          double hunter_y = std::stod(j[1]["hunter_y"].get<std::string>());
          double hunter_heading = std::stod(j[1]["hunter_heading"].get<std::string>());
          
          string lidar_measurment = j[1]["lidar_measurement"];
          
          MeasurementPackage meas_package_L;
          istringstream iss_L(lidar_measurment);
          long long timestamp_L;

          // reads first element from the current line
          string sensor_type_L;
          iss_L >> sensor_type_L;

      	  // read measurements at this timestamp
      	  meas_package_L.sensor_type_ = MeasurementPackage::LASER;
          meas_package_L.raw_measurements_ = VectorXd(2);
          float px;
      	  float py;
          iss_L >> px;
          iss_L >> py;
          meas_package_L.raw_measurements_ << px, py;
          iss_L >> timestamp_L;
          meas_package_L.timestamp_ = timestamp_L;          
          ukf.ProcessMeasurement(meas_package_L);
          string radar_measurment = j[1]["radar_measurement"];
            
          MeasurementPackage meas_package_R;
          istringstream iss_R(radar_measurment);
    	    long long timestamp_R;

          // reads first element from the current line
          string sensor_type_R;
          iss_R >> sensor_type_R;

      	  // read measurements at this timestamp
      	  meas_package_R.sensor_type_ = MeasurementPackage::RADAR;
          meas_package_R.raw_measurements_ = VectorXd(3);
          float ro;
      	  float theta;
      	  float ro_dot;
          iss_R >> ro;
          iss_R >> theta;
          iss_R >> ro_dot;
          meas_package_R.raw_measurements_ << ro,theta, ro_dot;
          iss_R >> timestamp_R;
          meas_package_R.timestamp_ = timestamp_R;
    	    ukf.ProcessMeasurement(meas_package_R);

          target_x = ukf.x_[0];
          target_y = ukf.x_[1];

          // std::cout << "px=" << ukf.x_[0] << ", py=" << ukf.x_[1] << std::endl;
          // std::cout << "v=" << ukf.x_[2] << ", phi=" << ukf.x_[3] << ", phid=" << ukf.x_[4] << std::endl;
          // std::cout << "R=" << ukf.x_[2] * delta_t / (2.*M_PI) << std::endl;

          double turn = 0.;
          bool chaseOn = false;
          // only started chasing when covariance is small enough i.e measurement is stable enough
          if(ukf.P_.maxCoeff() < 0.1) {
            double last_v = ukf.x_[2];
            double dt = 0.1;
            
            while (dt < 5.0) {              
              VectorXd x_pred;
              ukf.getPredictedState(x_pred, dt);

              double diff_x = x_pred[0] - hunter_x;
              double diff_y = x_pred[1] - hunter_y;
              if (sqrt(diff_x * diff_x + diff_y * diff_y) < dt * last_v) {
                chaseOn = true;
                //turn towards the target
                turn = atan2(diff_y, diff_x) - hunter_heading;
                while (turn > M_PI) turn-=2.*M_PI; 
                while (turn <-M_PI) turn+=2.*M_PI;

                break;
              }

              dt += 0.05;
            }

          } 
          
          // failed to chase, go back to simple head to current position
          if (!chaseOn){
            //turn towards the target
            turn = atan2(target_y - hunter_y, target_x - hunter_x) - hunter_heading;
            while (turn > M_PI) turn-=2.*M_PI; 
            while (turn <-M_PI) turn+=2.*M_PI;
          }

          double distance_difference = sqrt((target_y - hunter_y)*(target_y - hunter_y) 
                                            + (target_x - hunter_x)*(target_x - hunter_x));

          // std::cout << "turn=" << heading_difference << ", dist=" << distance_difference << "\n\n";

          json msgJson;
          msgJson["turn"] = turn;
          msgJson["dist"] = distance_difference; 
          auto msg = "42[\"move_hunter\"," + msgJson.dump() + "]";
          // std::cout << msg << std::endl;
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
	  
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }

  });

  // We don't need this since we're not using HTTP but if it's removed the program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1)
    {
      res->end(s.data(), s.length());
    }
    else
    {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, char *message, size_t length) {
    // ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port))
  {
    std::cout << "Listening to port " << port << std::endl;
  }
  else
  {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}























































































