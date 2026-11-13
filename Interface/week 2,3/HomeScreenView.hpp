#ifndef HOMESCREENVIEW_HPP
#define HOMESCREENVIEW_HPP

#include <gui_generated/homescreen_screen/HomeScreenViewBase.hpp>
#include <gui/homescreen_screen/HomeScreenPresenter.hpp>

class HomeScreenView : public HomeScreenViewBase
{
public:
    HomeScreenView();
    virtual ~HomeScreenView() {}
    virtual void setupScreen();
    virtual void tearDownScreen();
    virtual void addGraphData(int powerValue);
    virtual void updateApplianceUI(int ac, int fan, int light, int fridge);
    virtual void acDetailsClicked();
    virtual void fanDetailsClicked();
    virtual void lightDetailsClicked();
    virtual void fridgeDetailsClicked();
protected:
};

#endif // HOMESCREENVIEW_HPP
