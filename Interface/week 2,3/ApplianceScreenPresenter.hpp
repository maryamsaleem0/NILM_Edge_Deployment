#ifndef APPLIANCESCREENPRESENTER_HPP
#define APPLIANCESCREENPRESENTER_HPP
#include <cstdint>

#include <gui/model/ModelListener.hpp>
#include <mvp/Presenter.hpp>

using namespace touchgfx;

class ApplianceScreenView;

class ApplianceScreenPresenter : public touchgfx::Presenter, public ModelListener
{
public:
    ApplianceScreenPresenter(ApplianceScreenView& v);

    /**
     * The activate function is called automatically when this screen is "switched in"
     * (ie. made active). Initialization logic can be placed here.
     */
    virtual void activate();

    /**
     * The deactivate function is called automatically when this screen is "switched out"
     * (ie. made inactive). Teardown functionality can be placed here.
     */
    virtual void deactivate();

    virtual ~ApplianceScreenPresenter() {}
    uint8_t getSelectedAppliance();

private:
    ApplianceScreenPresenter();

    ApplianceScreenView& view;
};

#endif // APPLIANCESCREENPRESENTER_HPP
