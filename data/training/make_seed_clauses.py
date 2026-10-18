"""Hand-worded symptom sentences, one per line, grouped by label.

These are extra training sentences written independently of the wording in 1000_V2.csv,
so the classifier sees more than one way to say each fault. Run this file to (re)write
data/training/seed_clauses.csv.
"""
import collections
import csv
import os

S = {}
S["no_power"] = """unit is totally lifeless, nothing happens when I press start
won't come on at all
machine not starting, no indication on the panel
plugged in but there is no sign of life
it just will not boot up
appliance is not switching on since morning
no response at all when I plug it in
cx says unit not powering up
dead unit, no LED
will not start, tried two different sockets
it refuses to turn on
nothing lights up and nothing runs
shut down on its own and never came back on
not getting any power to the appliance
appliance is off and will not wake up
bilkul chalu nahi ho raha
on nahi ho raha hai, koi light nahi
start hi nahi hota
suddenly switched off and now it is not turning on
there is no power in the unit even though the plug is fine
does not turn on anymore
unit unresponsive, zero power
can't get it to switch on
it was working yesterday, today it is not coming on
power light is off and the machine does nothing
main switch is on but appliance stays lifeless
not turning on after the voltage came back
device is not starting up, no beep, no light
completely non functional, does not start
machine band pada hai, on nahi hota
nothing at all when I hit the on switch
unit not powering on, checked the outlet with another device
stays dark, will not power up
it died and will not restart
won't power up no matter what I do
not coming on, no standby light either"""
S["not_heating"] = """water is not getting hot
element is not warming up
food stays cold even after the full time
oven never reaches the set temperature
takes forever to heat
only lukewarm water comes out
heating is very weak
the grill is not getting hot
no hot water since two days
garam nahi ho raha
paani garam nahi hota
heats too slowly, an hour for a small tank
clothes come out damp because the dryer has no heat
the coil stays cold
temperature is way lower than what I set
cakes are coming out undercooked, oven runs cool
dishes are not dried, the heater does not seem to work
it runs but there is no heat at all
microwave runs and the plate turns but food is still cold
heater not working in heating mode, only cold air
water barely warm even on the highest setting
hob plate is not heating
does not warm the food
the heating function has stopped
cx reports no heating, unit otherwise running
not hot enough, thermostat set to max
oven is not baking properly, temperature seems too low
heat mode on the AC gives room temperature air
geyser se thanda paani aa raha hai
wash water remains cold on hot programs
preheating never finishes
overheats and burns everything, temperature is far too high
temperature is not right, it is either too hot or not hot at all
takes double the time to cook anything
no heat from the bottom element
the water does not heat on any program"""
S["not_cooling"] = """not cooling at all
thanda nahi kar raha
cooling bilkul nahi hai
the freezer is defrosting by itself and everything melted
vegetables are rotting, the fridge is not cold enough
room does not get cool even after hours
only blowing room temperature air
ice cream is soft, freezer not freezing
cooling is very poor
the inside feels warm to the touch
temperature inside keeps rising
frost has built up everywhere and it does not chill
AC is on but there is no chill
fridge compartment is warm while the freezer is fine
it stopped chilling
cooling goes away after some time
milk went sour because the fridge is warm
cx reports insufficient cooling
not maintaining low temperature
the coils are iced over and it barely cools
air from the vents is not cool
no cooling since the weekend
freezer is not making ice anymore
unit runs all day but the room stays hot
low cooling, thermostat at the coldest setting
bottles are not getting chilled
heavy frosting in the freezer and items in the fridge are warm
cooling kam ho gayi hai
everything in the freezer thawed
does not cool below 28 degrees
the air conditioner throws hot air
refrigerator is not cold inside
chiller section is warm
cooling drops in the afternoon
meat spoiled as the freezer got warm
weak cooling from the indoor unit"""
S["water_leak"] = """water on the floor around the machine
it leaks every time I run it
there is a pool of water below the unit
water keeps seeping out from the base
dripping from the pipe joint
paani leak ho raha hai
neeche se paani tapak raha hai
floor gets wet after each use
water is coming out from the bottom
leakage from the back side
water spilling out from the front during wash
the tank is leaking
overflowing and flooding the bathroom
drops of water falling from the indoor unit
wet patch under the appliance every day
water trickles down the wall from the AC
leaking badly, had to put a bucket
water escaping from the detergent drawer
there is a leak somewhere, the kitchen floor is soaked
cx reports water leakage from unit
steady drip from the underside
water comes out of the door area and runs across the floor
small leak near the inlet valve
it overflowed and water went everywhere
fridge is leaking water onto the floor
found water pooled beside the washer
water leaks out when it fills
the bottom tray overflows onto the floor
moisture and water dripping down the front
leak from the heater outlet
water oozing from the seam of the tank
soaked the carpet again, water leaking out
water running out from beneath
it is dribbling water constantly
puddles keep forming near it
pani bahar aa raha hai machine se"""
S["drainage_or_blockage"] = """water is not draining out
it will not pump the water out
tub stays full after the cycle ends
the outlet hose is choked
sink side drain is clogged and the machine backs up
dirty water sits at the bottom
filter is jammed with lint
paani nikal nahi raha
drain nahi ho raha hai
water stays inside and smells
pump is blocked by something
cannot empty, water level stays the same
drain line is clogged
the cycle ends but the water has not gone
waste water comes back into the tub
blockage in the drain, nothing flows out
takes ages to drain
the condensate line is clogged
water logged inside the drum
not emptying after rinse
the drain hole at the back of the fridge is choked
cx reports drain blocked, standing water in tub
filter fully blocked with debris
water does not go down the outlet
drainage problem, tub remains flooded
the hose is kinked and it cannot drain
stagnant water left inside after washing
food bits have clogged the drain
drain pump runs but water remains
it stops mid cycle full of water
unable to drain, shows water inside
water not going out through the pipe
the drain tray is full and the tube is blocked
drains very slowly and leaves water behind
clogged up, water backs up into the machine
waste pipe blocked, machine remains full"""
S["noise_or_vibration"] = """it is extremely noisy
makes a horrible racket
loud knocking while running
bahut awaaz karta hai
awaaz bahut zyada aa rahi hai
there is a constant whining sound
shakes the whole floor
thumping noise on every rotation
it rattles like something is loose
jumps around when spinning
high pitched screech at startup
clicking and clunking sounds all the time
a droning noise that keeps us awake
vibrates so much that things fall off the top
strange noise from the back
unusual sound, like metal scraping
unit is very loud compared to before
whistling noise from the indoor unit
rumbling sound when it runs
cx reports abnormal noise during operation
the machine dances across the room
excessive vibration on spin
making a ticking noise continuously
it roars like an engine
gurgling and knocking noises at night
loud hum, can hear it in the next room
it squeaks with every turn
noise level has increased a lot
keeps making a tapping sound
bangs against the wall while running
too much vibration, the counter shakes
harsh grating noise
loud thud each time it starts
sounds like a helicopter
the outdoor unit makes a rattling racket
jerks and bounces heavily"""
S["door_seal_or_latch"] = """door will not shut
the door does not stay closed
latch is broken
door lock is jammed
cannot open the door
the gasket is torn
rubber lining around the door has come loose
darwaza band nahi ho raha
door lock kharab hai
hinge snapped and the door hangs down
door pops open on its own
the handle came off and the door cannot be opened
lid will not close properly
door seal is cracked
the catch does not hold the door
door stuck shut, cannot get my laundry out
the door is misaligned and leaves a gap
rubber seal is mouldy and peeling off
door does not click into place
cx reports door latch faulty, door not closing
lock stays engaged after the cycle
the door swings open during use
freezer door does not seal
door is hard to close, have to slam it
latch hook broke off
lid lock is not releasing
oven door will not shut fully
the magnetic strip on the door is not sticking
door spring is broken
seal is damaged and the door does not close tight
door handle is loose and the latch does not catch
the door gets locked and will not unlock
microwave door will not latch
gasket hanging out of the door
door glass frame came apart and it does not close
door ka rubber nikal gaya hai"""
S["control_panel_or_display"] = """the screen is blank
keys are not responding
buttons do not work
touch controls are dead but the machine runs
display is flickering with random symbols
it forgets the time and the programs
panel is frozen
knob does not change the settings
the digits on the display are half missing
display kaam nahi kar raha
button dabane par kuch nahi hota
screen shows gibberish
timer display is stuck
cannot select any program, the panel does not react
the keypad is unresponsive
backlight of the display is gone
start button has to be pressed very hard to register
control panel lights are blinking randomly
the settings keep changing by themselves
remote sensor on the panel does not respond
display dim and unreadable
the selector dial spins freely and nothing changes
buttons stuck, only the power key works
cx reports display blank, unit functional
child lock icon is stuck on and the keys are locked
clock resets to zero every day
panel beeps but does not accept input
display shows wrong characters
touchscreen is not sensing my finger
the temperature readout is not showing
program selector is not working
indicator lights on the panel are not coming on
the display went black but the appliance still runs
some buttons work and some do not
the mode button is jammed
it loses the saved settings after every use"""
S["electrical_or_control_board"] = """power comes and goes on its own
keeps restarting by itself
the PCB has failed
main board is gone as per the engineer
it switches off and on randomly
turns off by itself in the middle and comes back
motherboard problem
lights flicker when it runs
power keeps cutting out
loses power for a few seconds every now and then
circuit board is damaged
the relay on the board keeps clicking and the unit resets
PCB kharab ho gaya hai
bar bar band chalu ho raha hai
electronic module needs replacing
the unit reboots itself again and again
random shutdowns throughout the day
supply to the unit is unstable, it flickers on and off
the control card is burnt out
cx reports intermittent power, suspected board failure
there was a short circuit inside
it gives a tingling sensation when touched
the wiring inside looks melted
goes off whenever the compressor starts and then comes back
power fluctuates inside the machine
the board got damaged after a voltage surge
keeps powering down unexpectedly
inverter board fault
works for a minute then cuts off and restarts
sometimes it has power and sometimes it does not
the electronics are fried
unit resets mid cycle
electrical fault in the control unit
the plug and cable get very hot
main PCB not working
it shuts off randomly and starts again later"""
S["motor_or_drive"] = """the drum is not rotating
motor does not run
fan is not working
compressor is not starting
the tub does not spin
blower has stopped
the plate inside does not turn
pump is not running
spin cycle does not work
motor hums but nothing moves
drum ghoom nahi raha
motor nahi chal rahi
fan nahi chal raha hai
belt seems broken, the drum turns freely by hand
agitator does not move
compressor runs for a second and stops
the fan blades are stuck
it only spins slowly
drum is jammed and will not turn
swing flap motor is not moving
the wash arms are not turning
outdoor unit fan stopped
compressor not kicking in
cx reports motor seized
spins in one direction only
the drive is slipping, no proper spin
fan runs at very low speed
the rotating tray is stuck
motor tries to start and gives up
exhaust fan of the oven does not come on
circulation motor is dead
the impeller is not turning
drum rotates weakly and stops under load
compressor is tripping on overload and never runs
blower speed does not change
the spinner is not working"""
S["installation_or_setup"] = """nobody has come to install it yet
installation is still pending
it was installed wrongly
not levelled properly
the machine is tilted to one side
inlet pipe was not attached
they forgot to fit the drain hose
does not fit in the cabinet
installation nahi hua abhi tak
fitting theek se nahi hui
demo and installation not done
the stand is uneven
mounting bracket is not fixed properly
outdoor unit has not been mounted
need help setting it up
the transit bolts were not removed
using it for the first time and do not know how to connect it
it came without the fittings
the gas connection is not done
plumbing connection is missing
wall mounting is crooked
just unboxed it and cannot set it up
the feet need adjusting, it is not sitting flat
installation team left the job half done
copper piping was not connected
wrong size for the slot, will not go in
cx requests installation, unit delivered last week
it is not fixed to the wall properly
setup is incomplete
first use after delivery and the hoses are not attached
the installer did not connect the water supply
needs to be levelled, it leans forward
electrical point for it was not set up by the fitter
anti-tip bracket not installed
the installer has not shown up
kitchen cutout is too small for it"""
S["none"] = """thank you
many thanks for your help
please get back to me soon
I am available after 6 pm
my address has changed
this is very frustrating
I have been a loyal customer for years
please do the needful
I bought it during the festival sale
the invoice number is attached
kindly escalate this to your manager
awaiting your reply
my order number is 48213
I work from home so anytime is fine
we have guests coming this weekend
it is a 7 kg model
I called the helpline twice already
your service is the best
very happy with the product
what is the price of the new model
do you have a showroom nearby
please share the brochure
I would like to become a dealer
limited time offer, call now
you have been selected for a prize
how do I register my product
can I extend my warranty
please cancel my service request
I want to change my appointment time
the technician was very helpful
good job by the team
jaldi kijiye please
dhanyavaad
bahut accha kaam kiya
is this model available in grey
please send the bill copy
where can I download the manual
want to know about exchange offers
I need a quotation for ten units
kindly confirm the visit time
it is under AMC
we bought it two years ago
it was a gift from my parents
I live on the third floor
last time the engineer came late
I do not want a refund
purchased online
serial number is on the back
hello
regards
looking forward to hearing from you
hope you are doing well
no issues at all, works perfectly
everything is fine now
please update my email id
unsubscribe me from your mailing list
visit our website for cheap loans
I am writing regarding my appliance
this is my third complaint
I am not happy with the support
I need this resolved before the weekend
the product is five months old
earlier it was working fine
my neighbour recommended your brand
please treat this as urgent
warranty card is with me
I have small children at home
send someone today itself
can you tell me the status of my request
when will the spare part arrive"""

# Second batch, written after an audit with fresh tickets showed where the first batch was thin:
# very short complaints, component-specific wording, Hinglish, and text that is not a fault
# (questions, resolved problems, negated symptoms, service complaints).
EXTRA = {}
EXTRA["no_power"] = """machine dead
dead
no power
not switching on
does not start
will not turn on since the power came back
totally dead, not even the display
machine chalu nahi ho rahi
chalu hi nahi hota
power nahi aa raha machine mein
unit is not coming on, no light, no sound
it stopped working completely and shows no sign of power
nothing happens when I plug it in and press the button
appliance has gone dead overnight
does not respond to the power switch at all
plugged in, switch on, still nothing
cx reports unit dead on arrival
the oven has no power at all
fridge is off, light inside is also off
AC does not turn on with remote or button
washer is dead, cannot start any program
it will not power on after the storm
no display, no light, no sound, fully dead
band ho gaya aur ab chalu nahi ho raha"""
EXTRA["not_heating"] = """not heating
no hot water
geyser water not hot
water not hot
oven not heating properly
takes too long to heat
heater garam nahi kar raha
garam paani nahi aa raha
geyser garam paani nahi de raha
khana garam nahi hota microwave mein
the gas burner will not ignite, only clicking
burner does not light, there is no flame
the flame keeps going out on the burner
pilot light is out and the water stays cold
one ring on the hob stays cold
grill element does not glow
it heats for a minute and then goes cold
food is still frozen in the middle after the full time
water is warm at first and then turns cold quickly
the oven stays cold even at the highest setting
dryer runs but clothes stay wet, no warm air
cx reports no heat from unit
rods are not heating up
bottom of the cake stays raw, the oven is too cool"""
EXTRA["not_cooling"] = """not cooling
ac not cooling
fridge not cooling
no cooling
fridge thanda nahi ho raha hai
AC thandi hawa nahi de raha
thanda nahi karta
freezer mein baraf nahi jam rahi
the compressor runs all the time but the freezer is warm
it runs continuously and still the inside is warm
compressor is running but there is no cooling
the room stays warm although the AC runs for hours
ice is melting in the freezer
the fridge feels warm and the butter is soft
fan blows but the air is not cold
cooling was fine yesterday, today it is warm inside
only the top shelf is cold, the rest is warm
it trips on cooling, the coil is covered in frost
the back wall is iced up and the food is warm
gas has leaked from the AC and it does not cool
cx c/o unit not cooling
drinks are not cold after a full day
freezer works but the fridge section is warm
temperature shows 15 degrees and will not go down"""
EXTRA["water_leak"] = """leaking
water leaking
leak from bottom
water leaking from bottom
machine se paani gir raha hai
paani gir raha hai neeche
paani tapak raha hai
fridge se paani nikal raha hai
paani beh raha hai machine ke neeche se
water drips from the indoor unit onto the bed
the hose joint is leaking
there is always a wet floor under the dishwasher
water is leaking from the door during the wash
tank is dripping from the valve
a puddle forms under the fridge every night
it leaks from the soap drawer
water came out from under the machine and spread over the floor
the pipe at the back is leaking
cx reports leakage from bottom of unit
slow leak, the cabinet floor is damp
water drops from the AC inside the room
the geyser is leaking from the top
leak near the drain connection, water on the tiles
water leaks whenever it fills up"""
EXTRA["drainage_or_blockage"] = """not draining
water not draining
drain blocked
water stays in the drum
paani nahi nikal raha machine se
paani bahar nahi ja raha
drain jam ho gaya hai
there is no leak, but the machine does not drain
no leak, but the water does not go out
the water does not pump out at the end
DW not draining, standing water
standing water in the bottom after the cycle
the machine is full of water and will not empty
dirty water is left in the tub
the drain pipe is choked with lint
filter is clogged and the water stays inside
it shows a drain error and the water remains
water backs up into the sink when it drains
cx reports unit not draining
the outlet is blocked, nothing comes out of the hose
water collects inside the fridge because the drain hole is blocked
the AC drain line is blocked and the tray is full
slow drain, takes very long to empty
rinse water stays behind every time"""
EXTRA["noise_or_vibration"] = """noisy
very noisy
loud noise
making noise
making loud noise
AC se bahut awaz aa rahi hai
bahut awaz aa rahi hai
awaz bahut hai
machine bahut awaz karti hai
khat khat ki awaz aati hai
bahut vibrate karta hai
noise during spin
there was no noise before, now it is very loud
it hums loudly all night
buzzing sound from the back
it makes a knocking sound when it starts
a rattling sound comes from inside
the fan makes a scraping sound
vibration is so strong the shelf shakes
it shakes a lot during the spin
cx reports abnormal sound from compressor
loud clicking every few seconds
whirring sound that gets louder
grinding noise when the drum turns"""
EXTRA["door_seal_or_latch"] = """door not closing
door not locking
door will not open
door stuck
darwaza lock nahi ho raha
darwaza nahi khul raha
door band nahi hota
dhakkan band nahi ho raha
the door seal is torn
door gasket has come off
the latch does not click
the door pops open during the cycle
fridge door does not shut properly, it stays slightly open
the lid does not lock so it will not start the spin
door handle is broken and the door cannot be pulled open
door remains locked after the wash is over
the rubber around the door is damaged
hinge is loose and the door sags
cx reports door not latching
oven door does not close flush
the microwave door catch is broken
freezer door seal is not sticking
the door lock is jammed
door swings open by itself"""
EXTRA["control_panel_or_display"] = """display blank
display not working
buttons not working
showing error
display is showing error
it shows an error
error on the display
display par error aa raha hai
display band hai
button kaam nahi kar rahe
remote does not work and the display on the unit is off
the display is off but it still runs
touch buttons do not respond
panel lights are off although the machine works
the screen is dead but the fan is running
display shows strange letters
the timer does not count down on the display
the knob turns but the setting does not change
half the display is not lit
settings reset themselves every time
cx reports control panel unresponsive
the keys beep but nothing is selected
the digital display flickers and goes blank
cannot set the temperature, the buttons are stuck"""
EXTRA["electrical_or_control_board"] = """power keeps going off and on
it switches off by itself
keeps restarting
board problem
PCB issue
pcb fault
bar bar band ho jata hai
apne aap band ho jata hai
kabhi chalta hai kabhi nahi
the technician says the main board has failed
it works for some time and then cuts off on its own
it turns itself off after a few minutes and restarts
the unit resets in the middle of the program
power drops out randomly
lights on the unit flicker
the control board is faulty
main PCB needs to be replaced
it loses power for a moment and comes back
cx reports intermittent shutdown
the supply to the machine is not steady
the plug gets hot when the machine runs
it shuts down whenever the heater comes on and restarts later
the electronic board got damaged in the voltage fluctuation
power comes for a second and goes"""
EXTRA["motor_or_drive"] = """not spinning
drum not spinning
fan not running
compressor not starting
motor not working
microwave plate not rotating
plate not rotating
drum nahi ghoom raha
pankha nahi chal raha
motor nahi chalti
compressor chalu nahi ho raha
clothes are still wet after the spin cycle
it does not spin, the clothes come out dripping
dishes come out dirty and the arms are not turning
the spray arms are not turning
the wash arms are not rotating
the turntable does not rotate
the drum is stuck and will not move
fan inside does not rotate
the compressor does not start, it only clicks
blower is not working, no air comes out
cx reports drum not rotating
the pump motor does not run
it tries to spin and then stops"""
EXTRA["installation_or_setup"] = """installation pending
not installed yet
installation not done
installation nahi hua
abhi tak install nahi kiya
technician install karne nahi aaya
where do I connect the inlet pipe
how do I connect the drain hose
just delivered and I do not know where to connect the pipe
bought last week and it wobbles a lot while spinning
the new machine wobbles because it is not level
it rocks on the floor, the legs are not adjusted
the oven does not fit in my kitchen slot
it does not fit in the space
the unit is too wide for the gap
first time I am using it and it is not set up properly
the stand was not fitted
the outdoor unit is not fixed properly on the bracket
pipe connection is loose after installation
the installer left without finishing the job
it was not levelled during installation
cx requests installation and demo
wrong fitting was used for the gas pipe
the machine is not placed properly and tilts"""
EXTRA["none"] = """how do I clean the filter
how do I use the timer
how to set the clock
what does the eco mode do
what is the capacity of this model
can I wash woollens in it
how often should I defrost it
which detergent should I use
my order number is 48213, when will it be delivered
when will my order be delivered
where is my order
please share the delivery date
I want a refund
I want to return the product
please cancel my order
your technician did not come today
the technician did not come
nobody came for the visit
no one has called me back
I have been waiting for three days
your customer care is not picking up
the delivery guy was rude
kindly share the warranty terms
is it still under warranty
please share the invoice
I need the service centre address
please call me
call me back
hello
hi
ok thanks
thank you for fixing it
it is working fine now
the fridge is cooling fine now, thank you for fixing it
cooling is fine now
there is no leak
no leak
there is no noise
no noise at all
it does not make any noise
it does not leak
no problem with the door
the display is fine
it was fine until the power cut
there was a power cut yesterday
power cut in our area
we had a power cut last night
asdfgh qwerty
12345
test
please ignore
sent by mistake
thik hai dhanyavaad
bahut badhiya service
jaldi aaiye
kab aayega technician
kitna charge lagega
price kya hai
I am very angry
this is unacceptable
I purchased it in 2019 from the local store
we bought this model two years ago
the model number is on the label and the serial number is below it
I was baking at 400 degrees for ten minutes
I was cooking dinner when it happened
we were washing a normal load of clothes
I had set it to the normal cycle before going to bed
the oven had been on for about an hour
we use it only for normal cooking
no one was injured
luckily nobody was hurt
I contacted the manufacturer and they refused to help
the company said it is out of warranty
they told me it is not part of the recall
a technician came out twice and replaced a part
the repairman said he had seen this before
I paid for the repair myself
I had to take a day off work for the visit
I found many similar complaints online
this product should be recalled
I have photos if you need them
we have two small children in the house
I am worried it could hurt someone
there is mould and a musty smell inside
black mold has grown inside the unit
it smells of mildew after every wash
the sharp edge cut my hand
I cut my finger on the metal fins
the rack slid out and the dish fell
it has caused allergies and coughing
we had to throw away all the food
the food spoiled and we got sick
I would like a refund or a replacement
I want the manufacturer to be held accountable
I filed a complaint with the retailer
the customer service line kept me on hold
we moved into this house last year
it came with the house when we bought it"""


if __name__ == "__main__":
    rows = [(t.strip(), lab) for lab, block in list(S.items()) + list(EXTRA.items()) for t in block.split("\n") if t.strip()]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed_clauses.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(rows)
    print(len(rows), "rows ->", out)
    print(dict(collections.Counter(lab for _, lab in rows)))
