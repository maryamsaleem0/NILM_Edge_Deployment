#ifndef HOMESCREENPRESENTER_HPP
#define HOMESCREENPRESENTER_HPP
#include <cstdint>
#include <gui/model/ModelListener.hpp>
#include <mvp/Presenter.hpp>

using namespace touchgfx;

class HomeScreenView;

class HomeScreenPresenter : public touchgfx::Presenter, public ModelListener
{
public:
    HomeScreenPresenter(HomeScreenView& v);

    /**
     * The activate function is called automatically when this screen is "switched in"
     * (ie. made active). Initialization logic can be placed here.
     */
    virtual void updatePowerGraph(int powerValue);
    virtual void activate();
    virtual void updateAppliances(int ac, int fan, int light, int fridge);

    /**
     * The deactivate function is called automatically when this screen is "switched out"
     * (ie. made inactive). Teardown functionality can be placed here.
     */
    virtual void deactivate();

    virtual ~HomeScreenPresenter() {}
    void setSelectedAppliance(uint8_t id);
private:
    HomeScreenPresenter();

    HomeScreenView& view;
};

#endif // HOMESCREENPRESENTER_HPP
