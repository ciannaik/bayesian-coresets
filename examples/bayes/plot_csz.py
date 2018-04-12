import numpy as np
import bokeh.plotting as bkp
import bokeh.io as bki
import bokeh.layouts as bkl
from bokeh.models import FuncTickFormatter
import bokeh.models as bkm
import bokeh.palettes 
import time

logFmtr = FuncTickFormatter(code="""
var trns = [
'\u2070',
'\u00B9',
'\u00B2',
'\u00B3',
'\u2074',
'\u2075',
'\u2076',
'\u2077',
'\u2078',
'\u2079'];
var tick_power = Math.floor(Math.log10(tick));
var tick_mult = Math.pow(10, Math.log10(tick) - tick_power);
var ret = '';
if (tick_mult > 1.) {
  if (Math.abs(tick_mult - Math.round(tick_mult)) > 0.05){
    ret = tick_mult.toFixed(1) + '\u22C5';
  } else {
    ret = tick_mult.toFixed(0) +'\u22C5';
  }
}
ret += '10';
if (tick_power < 0){
  ret += '\u207B';
  tick_power = -tick_power;
}
power_digits = []
while (tick_power > 9){
  power_digits.push( tick_power - Math.floor(tick_power/10)*10 )
  tick_power = Math.floor(tick_power/10)
}
power_digits.push(tick_power)
for (i = power_digits.length-1; i >= 0; i--){
  ret += trns[power_digits[i]];
}
return ret;
""")



#dnames = ['poiss/synth', 'poiss/biketrips', 'poiss/airportdelays', 'lr/synth', 'lr/ds1', 'lr/phishing']
dnames = ['poiss/synth']
fig_csz = bkp.figure(y_axis_type='log', y_axis_label='Normalized Fisher Information Distance', x_axis_type='log', x_axis_label='Coreset Size', x_range=(.05, 1.1), plot_width=1250, plot_height=1250)

axis_font_size='36pt'
legend_font_size='36pt'
#fig_csz.xaxis.ticker = bkm.tickers.FixedTicker(ticks=[.1, .5, 1])
fig_csz.xaxis.axis_label_text_font_size= axis_font_size
fig_csz.xaxis.major_label_text_font_size= axis_font_size
fig_csz.xaxis.formatter = logFmtr
fig_csz.yaxis.axis_label_text_font_size= axis_font_size
fig_csz.yaxis.major_label_text_font_size= axis_font_size
fig_csz.yaxis.formatter = logFmtr
#fig_csz.toolbar.logo = None
#fig_csz.toolbar_location = None


pal = bokeh.palettes.colorblind['Colorblind'][8]
pal = [pal[0], pal[1], '#d62728', pal[3], pal[4], pal[5], pal[6], pal[7], pal[2]]
for didx, dnm in enumerate(dnames):
  
  res = np.load(dnm  + '_results.npz')

  Fs = res['Fs']
  Fs_full = res['Fs_full']
  cputs = res['cputs']
  cputs_full = res['cputs_full']
  csizes = res['csizes']
  anms = res['anms']

  for aidx, anm in enumerate(anms):
    anm = anm.decode('utf-8')
    if anm == 'FW':
      clr = pal[1]
    elif anm == 'GIGA':
      clr = pal[0]
    else:
      clr = pal[2]

    fig_csz.line(np.percentile(csizes[aidx,:,:], 50, axis=0), np.percentile(Fs[aidx, :, :], 50, axis=0)/np.percentile(Fs[2, :, :], 50), line_color=clr, line_width=8, legend=anm)
    
fig_csz.legend.label_text_font_size= legend_font_size
fig_csz.legend.glyph_width=40
fig_csz.legend.glyph_height=80
fig_csz.legend.spacing=20
fig_csz.legend.orientation='horizontal'

bkp.show(fig_csz)

