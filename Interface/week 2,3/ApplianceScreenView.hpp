#ifndef APPLIANCESCREENVIEW_HPP
#define APPLIANCESCREENVIEW_HPP

#include <gui_generated/appliancescreen_screen/ApplianceScreenViewBase.hpp>
#include <gui/appliancescreen_screen/ApplianceScreenPresenter.hpp>

class ApplianceScreenView : public ApplianceScreenViewBase
{
public:
    ApplianceScreenView();
    virtual ~ApplianceScreenView() {}
    virtual void setupScreen();
    virtual void tearDownScreen();
protected:
};

#endif // APPLIANCESCREENVIEW_HPP
