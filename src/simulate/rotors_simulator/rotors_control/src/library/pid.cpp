#include <iostream>
#include <cmath>
#include "rotors_control/pid.h"

using namespace std;

class PIDImpl
{
public:
    PIDImpl(double dt, double max, double min, double Kp, double Kd, double Ki, 
            double alphaP, double alphaD, double antiwindup_radius, double integrator_max);
    ~PIDImpl();
    double calculate(double setpoint, double pv);
    void update_gains(double P, double D, double I, double alphaP, double alphaD);
    void reset();

private:
    double _dt;
    double _max;
    double _min;
    double _Kp;
    double _Kd;
    double _Ki;
    double _antiwindup_radius;
    double _integrator_max;
    double _error_prev = 0.0;
    double _integral = 0.0;
    double _alphaP = 0.0;
    double _alphaD = 0.0;
    double _Dout_pre = 0.0;
    double _Pout_pre = 0.0;
};

PID::PID(double dt, double max, double min, double Kp, double Kd, double Ki, 
        double alphaP, double alphaD, double antiwindup_radius, double integrator_max)
{
    pimpl = new PIDImpl(dt, max, min, Kp, Kd, Ki, alphaP, alphaD, antiwindup_radius, integrator_max);
}
double PID::calculate(double setpoint, double pv)
{
    return pimpl->calculate(setpoint, pv);
}

void PID::update_gains(double P, double D, double I, double alphaP, double alphaD)
{
    return pimpl->update_gains(P, D, I, alphaP, alphaD);
}
void PID::reset()
{
    return pimpl->reset();
}

PID::~PID()
{
    delete pimpl;
}

PIDImpl::PIDImpl(double dt,
                 double max, double min,
                 double Kp, double Kd, double Ki,
                 double alphaP, double alphaD, 
                 double antiwindup_radius, double integrator_max) : _dt(dt),
                                 _max(max),
                                 _min(min),
                                 _Kp(Kp),
                                 _Kd(Kd),
                                 _Ki(Ki),
                                 _antiwindup_radius(antiwindup_radius),
                                 _integrator_max(integrator_max),
                                 _error_prev(0),
                                 _integral(0),
                                 _alphaP(alphaP),
                                 _alphaD(alphaD)
{
}

void PIDImpl::update_gains(double P, double D, double I, double alphaP, double alphaD)
{
    _Kp = P;
    _Kd = D;
    _Ki = I;
    _alphaP = alphaP;
    _alphaD = alphaD;
}
double PIDImpl::calculate(double setpoint, double pv)
{

    // Error
    double error = setpoint - pv;
    // std::cout << "error = " << error << endl;
    // Proportional
    double Pout_tmp = _Kp * error;
    double Pout = _alphaP * _Pout_pre + (1 - _alphaP) * Pout_tmp;

    // Integral
    if (std::abs(error) < _antiwindup_radius) // only integrate when the error is small
    {
        _integral += error * _dt;
    }
    if (_integral > _integrator_max)
        _integral = _integrator_max;
    else if (_integral < -_integrator_max)
        _integral = -_integrator_max;
    double Iout = _Ki * _integral;

    // Derivative
    double derivative = (error - _error_prev) / _dt;
    double Dout_tmp = _Kd * derivative;
    double Dout = _alphaD * _Dout_pre + (1 - _alphaD) * Dout_tmp;

    // Calculate total output
    double output = Pout + Iout + Dout;

    // Check bounds
    if (output > _max)
        output = _max;
    else if (output < _min)
        output = _min;

    // Swap errors
    _error_prev = error;
    _Dout_pre = Dout;
    _Pout_pre = Pout;

    return output;
}

void PIDImpl::reset()
{
    _integral = 0.0;
    _error_prev = 0.0;
    _Dout_pre = 0.0;
    _Pout_pre = 0.0;
}

PIDImpl::~PIDImpl()
{
}