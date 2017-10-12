#ifndef PID_H
#define PID_H

class PID {
private:
  /*
  * Errors
  */
  double p_error_;
  double i_error_;
  double d_error_;

  /*
  * Coefficients
  */ 
  double Kp_;
  double Ki_;
  double Kd_;

 /*
  * Counter & previous cte
  */
  int    nb_update_;
  double sum_sqr_err_;
  double prev_cte_;

public:  
  /*
  * Constructor
  */
  PID();

  /*
  * Destructor.
  */
  virtual ~PID();

  /*
  * Initialize PID.
  */
  void Init(double Kp, double Ki, double Kd);

  /*
  * Update the PID error variables given cross track error.
  */
  void UpdateError(double cte);

  /*
  * Calculate the total PID error.
  */
  double TotalError();

  /*
   * Print debug line
   */
  void PrintDebug() const;
};

#endif /* PID_H */
